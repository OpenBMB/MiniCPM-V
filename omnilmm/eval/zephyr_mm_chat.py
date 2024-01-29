import os
import copy

import torch

from PIL import Image
from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM

from omnilmm.eval.muffin_vqa import expand_question_into_multimodal, KeywordsStoppingCriteria
from omnilmm.train.train_utils import _add_speaker_and_signal, _tokenize_fn
from omnilmm import conversation as conversation_lib
from omnilmm.utils import disable_torch_init
from omnilmm.model.zephyr_mm import ZephyrMMForCausalLM
from omnilmm.model.utils import build_transform
from omnilmm.train.train_utils import zephyr_preprocess

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def init_zephyr_mm(model_path, device=None, tune_clip=False):
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load zephyr_mm model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=2048)

    model = ZephyrMMForCausalLM.from_pretrained(
        model_name, tune_clip=tune_clip).to(device='cuda', dtype=torch.bfloat16)
    image_processor = build_transform(
        is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP')

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert mm_use_im_start_end

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IM_END_TOKEN], special_tokens=True)

    if tune_clip:
        vision_tower = model.model.vision_tower
    else:
        vision_tower = model.model.vision_tower[0]
    resampler = model.model.resampler
    dtype = torch.bfloat16
    if device is not None:
        vision_tower.to(device=device, dtype=dtype)
        model.to(device=device, dtype=dtype)
    else:
        vision_tower.to(device='cuda', dtype=dtype)
        model.to(device='cuda', dtype=dtype)

    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer


def wrap_question_for_zephyr_mm(question, image_token_len, tokenizer):
    question = expand_question_into_multimodal(
        question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

    conversation = [
        {
            'role': 'user',
            'content': question
        }
    ]
    data_dict = zephyr_preprocess(sources=[conversation],
                                  tokenizer=tokenizer,
                                  generation=True)

    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])
    return data_dict


def wrap_question_for_zephyr(question, tokenizer):
    conversation = [
        {
            'role': 'user',
            'content': question
        }
    ]
    data_dict = zephyr_preprocess(sources=[conversation],
                                  tokenizer=tokenizer,
                                  generation=True)

    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])
    return data_dict


class ZephyrMMForSingleTurnChat:
    def __init__(self, model, img_processor, image_token_len, tokenizer) -> None:
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer

    def llm_decode(self, input_ids):
        with torch.inference_mode():
            input_size = input_ids.shape[-1]

            output = self.model.generate(
                input_ids=input_ids.unsqueeze(0).cuda(),
                temperature=1.2,
                max_new_tokens=1024,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=50,
                top_p=0.95,
            )

            response = self.tokenizer.decode(
                output.sequences[0][input_size:], skip_special_tokens=True)
            response = response.strip()
            return response

    def decode(self, image, input_ids):
        with torch.inference_mode():
            num_beams = 3
            input_size = input_ids.shape[-1]
            print(f'Input: {self.tokenizer.batch_decode(input_ids)}'
                  f'input_ids: {input_ids}')

            output = self.model.generate(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=1.2,
                max_new_tokens=1024,
                # num_beams=num_beams,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=50,
                top_p=0.95,
                # length_penalty=1.0
            )

            response = self.tokenizer.decode(
                output.sequences[0][input_size:], skip_special_tokens=True)
            # print(f'raw response is {response}')
            response = response.strip()
            return response

    def chat(self, image_path, question):
        image = Image.open(image_path).convert('RGB')
        input_ids = wrap_question_for_zephyr_mm(
            question, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)
        image = self.image_transform(image)

        return self.decode(image, input_ids)

    def llm_chat(self, question):
        input_ids = wrap_question_for_zephyr(
            question, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)

        return self.llm_decode(input_ids)


class ZephyrMMForMultiTurnChat(ZephyrMMForSingleTurnChat):
    def __init__(self, model, img_processor, image_token_len, tokenizer) -> None:
        super(ZephyrMMForMultiTurnChat, self).__init__(
            model, img_processor, image_token_len, tokenizer)
        self.history = []
        self.image = None

    def _update_history(self, question, out):
        self.history.append({
            'role': 'user',
            'content': question
        })
        self.history.append({
            'role': 'assistant',
            'content': out
        })

    def start_chat(self, image_path, question):
        image = Image.open(image_path).convert('RGB')
        input_ids = wrap_question_for_zephyr_mm(
            question, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)
        image = self.image_transform(image)

        out = self.decode(image, input_ids)

        question_with_image = expand_question_into_multimodal(
            question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

        self._update_history(question_with_image, out)
        self.image = image
        return out

    def resume(self, question):
        if self.image is None or len(self.history) == 0:
            print(f'Please first start chat before resuming.')
            return ''
        conv = copy.deepcopy(self.history) + [{
            'role': 'user',
            'content': question
        }]
        input_ids = zephyr_preprocess(sources=[conv],
                                    tokenizer=tokenizer,
                                    generation=True)['input_ids'][0]

        out = self.decode(self.image, input_ids)
        self._update_history(question, out)
        return out

    def clear(self):
        self.history = []
        self.image = None


if __name__ == '__main__':
    tune_clip = True
    model, img_processor, image_token_len, tokenizer = init_zephyr_mm(
        '/home/yutianyu/Zephyr_checkpoints/SFT_exp/zephyr_mm_12b_SFT-6node_5kPT_SFT_stage3-caterpillar-stage2_mix#caterpillar-stage3_lvis#caterpillar-stage3_svit#caterpillar-stage3_sharegpt4v#llava#unimm-chat-134#222#600#101#157#117/checkpionts/checkpoint-4000', tune_clip=tune_clip)
    chat_model = ZephyrMMForSingleTurnChat(
        model, img_processor, image_token_len, tokenizer)
    chat_model = ZephyrMMForMultiTurnChat(
        model, img_processor, image_token_len, tokenizer)
