import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import io
import json
import pathlib
import base64
import glob
from tqdm import tqdm
import shortuuid

from muffin import Beit3LlavaLlamaForCausalLM
from muffin.conversation import conv_templates
from muffin.utils import disable_torch_init
from muffin.model.utils import build_transform
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math


def load_img(img_path):
    try:
        with open(img_path, "rb") as f:
            img = f.read()
    except Exception as e:
        print(e, )
        img = "NO IMG"
    return img


def is_yes(text):
    if 'yes' in text.lower():
        return True
    else:
        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(
            f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f'Load beit3 model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        model = Beit3LlavaLlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16).cuda()
        image_processor = build_transform(
            is_train=False, input_size=model.model.vision_tower.args.img_size)

        mm_use_im_start_end = getattr(
            model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower
        vision_tower.to(device='cuda', dtype=torch.float16)
        vision_config = model.model.vision_config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = model.model.config.num_query
    else:
        raise NotImplementedError
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16).cuda()

        vision_tower = CLIPVisionModel.from_pretrained(
            args.vision_tower, torch_dtype=torch.float16).cuda()
        image_processor = build_transform(is_train=False)

        mm_use_im_start_end = getattr(
            model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size //
                           vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(
            vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(
            args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict(
            {k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    if not pathlib.Path(args.out_dir).exists():
        pathlib.Path(args.out_dir).mkdir(parents=True)

    for task_path in list(glob.glob(f'{args.mme_dir}/*')):
        task_name = task_path.split('/')[-1]

        if pathlib.Path(f'{args.out_dir}/{task_name}.txt').exists():
            continue
        ans_file_path = f'{args.out_dir}/{task_name}.txt'
        ans_file = open(ans_file_path, "w", encoding='utf-8')
        print(f'Write to {ans_file_path}')

        raw_ans_file = open(ans_file_path + '.jsonl', "w", encoding='utf-8')

        yes_cnt = 0
        no_cnt = 0
        for sample_path in list(glob.glob(f'{task_path}/*.txt')):
            # print(sample_path, flush=True)
            lines = open(sample_path, encoding='utf-8').readlines()
            image_file = sample_path.replace('.txt', '.jpg')
            if not pathlib.Path(image_file).exists():
                image_file = sample_path.replace('.txt', '.png')

            for line in lines:
                qs = line.split('\t')[0]
                if args.CoT:
                    qs += ' You should first explain how to get the answer step-by-step and finally give the answer in one word.'
                cur_prompt = qs
                if mm_use_im_start_end:
                    qs = qs + '\n' + DEFAULT_IM_START_TOKEN + \
                        DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
                else:
                    qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

                if args.conv_mode == 'simple_legacy':
                    qs += '\n\n### Response:'
                # conv = default_conversation.copy()
                conv = conv_templates[args.conv_mode].copy()

                # FIX LLaVA bug (inconsistent prompt between training and inference)
                conv.messages = []
                conv.sep = '\n###'

                conv.append_message(conv.roles[0], qs)
                prompt = conv.get_prompt()
                prompt = prompt + ' Assistant:'
                inputs = tokenizer([prompt])

                if len(image_file) > 1000:  # HACK: process b64 io stream
                    image = Image.open(io.BytesIO(
                        base64.b64decode(image_file))).convert('RGB')
                else:
                    image = Image.open(os.path.join(
                        args.image_folder, image_file)).convert('RGB')
                # image.save(os.path.join(save_image_folder, image_file))
                image_tensor = image_processor(image)

                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                # new stopping implementation
                class KeywordsStoppingCriteria(StoppingCriteria):
                    def __init__(self, keywords, tokenizer, input_ids):
                        self.keywords = keywords
                        self.tokenizer = tokenizer
                        self.start_len = None
                        self.input_ids = input_ids

                    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                        if self.start_len is None:
                            self.start_len = self.input_ids.shape[1]
                        else:
                            outputs = self.tokenizer.batch_decode(
                                output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                            for keyword in self.keywords:
                                if keyword in outputs:
                                    return True
                        return False

                keywords = ['###']
                stopping_criteria = KeywordsStoppingCriteria(
                    keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    # print(f'Inference with prompt=>{tokenizer.batch_decode(input_ids)}<=')
                    output = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        temperature=0.7,
                        max_new_tokens=20,
                        num_beams=3,
                        output_scores=True,
                        return_dict_in_generate=True,
                        stopping_criteria=[stopping_criteria],
                        repetition_penalty=1.1)
                output_ids = output.sequences
                score = output.scores
                # print(f'Scores: {score[0].topk(10)}')

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (
                    input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(
                        f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(
                    output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                # print(f'Output: {output_ids}@{tokenizer.batch_decode(output_ids[:, input_token_len:])}@{outputs}@')

                if args.conv_mode == 'simple_legacy' or args.conv_mode == 'simple':
                    while True:
                        cur_len = len(outputs)
                        outputs = outputs.strip()
                        for pattern in ['###', 'Assistant:', 'Response:']:
                            if outputs.startswith(pattern):
                                outputs = outputs[len(pattern):].strip()
                        if len(outputs) == cur_len:
                            break

                try:
                    index = outputs.index(conv.sep)
                except ValueError:
                    outputs += conv.sep
                    index = outputs.index(conv.sep)

                outputs = outputs[:index].strip()
                # raw_ans_file.write(outputs + '\t' + base64.b64encode(load_img(os.path.join(args.image_folder, image_file))).decode('utf-8') + '\n')
                raw_ans_file.write(outputs + '\t' + image_file + '\n')
                raw_ans_file.flush()
                if is_yes(outputs):
                    outputs = 'Yes'
                    yes_cnt += 1
                else:
                    if 'no,' not in outputs.lower() and 'no.' not in outputs.lower() and outputs.lower() != 'no':
                        print(f'=> Output not yes/no: {outputs}', flush=True)
                    outputs = 'No'
                    no_cnt += 1

                out_line = f'{image_file.split("/")[-1]}\t{line.strip()}\t{outputs}\n'
                # print(out_line, flush=True)
                ans_file.write(out_line)
                ans_file.flush()
        print(f'{task_name}: Yes={yes_cnt}, No={no_cnt}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--mme_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="default")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument('--CoT', default=0, type=int)
    args = parser.parse_args()

    print(f'@===> MME CoT: {args.CoT} <===@')
    eval_model(args)
