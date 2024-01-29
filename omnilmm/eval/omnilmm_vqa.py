import os
import json
import torch
import argparse

import torch.utils.data as torch_data

import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoConfig

from omnilmm import Beit3LlavaLlamaForCausalLM
from omnilmm.conversation import conv_templates
from omnilmm.utils import disable_torch_init
from omnilmm.model.utils import build_transform
from transformers import StoppingCriteria
from omnilmm.data.datasets import MultimodalQADataset


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


def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text:
        question_text = question_text.replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
    else:
        question_text = im_st_token + im_patch_token * \
            image_token_len + im_ed_token + '\n' + question_text
    return question_text


def wrap_question_with_default_conv(question_text, image_token_len):
    question_text = expand_question_into_multimodal(
        question_text, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)
    conv = conv_templates['default'].copy()
    conv.messages = []
    conv.sep = '\n###'

    conv.append_message(conv.roles[0], question_text)
    prompt = conv.get_prompt()
    return prompt


def torch_pad_sequence(sequence, padding_value, batch_first=True, padding_side='right'):

    if padding_side == 'right':
        sequence = torch.nn.utils.rnn.pad_sequence(
            sequence,
            batch_first=batch_first,
            padding_value=padding_value)
    elif padding_side == 'left':
        sequence = torch.nn.utils.rnn.pad_sequence(
            [v.flip(-1) for v in sequence],
            batch_first=batch_first,
            padding_value=padding_value)
        sequence = sequence.flip(-1)
    else:
        raise NotImplementedError(f'padding_size={padding_side}')
    return sequence


def qa_colloator_fn(data_list, tokenizer, img_transform):
    questions = [x['question'] for x in data_list]
    tokenized = tokenizer(questions)

    input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
    input_ids = torch_pad_sequence(
        input_ids, tokenizer.pad_token_id, padding_side='left')

    attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
    attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')

    images = [img_transform(x['image']) for x in data_list]
    images = torch.stack(images)

    raw_questions = [x['raw_question'] for x in data_list]
    data = {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'raw_questions': raw_questions
    }

    if 'question_id' in data_list[0]:
        data['question_id'] = [x['question_id'] for x in data_list]

    return data


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_size):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.input_size = input_size

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for o in output_ids:
            o = self.tokenizer.decode(
                o[self.input_size:], skip_special_tokens=True)
            if all([keyword not in o for keyword in self.keywords]):
                return False
        return True


def init_omnilmm(model_path, device=None):
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load omnilmm model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    patch_config(model_name)
    model = Beit3LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda()
    image_processor = build_transform(
        is_train=False, input_size=model.model.vision_tower.args.img_size)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.model.vision_tower
    if device is not None:
        vision_tower.to(device=device, dtype=torch.float16)
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)

    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer


def eval_model(args):
    model, image_processor, image_token_len, tokenizer = init_omnilmm(
        args.model_name)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    qa_dataset = MultimodalQADataset(args.question_file, partial(
        wrap_question_with_default_conv, image_token_len=image_token_len))

    collate_fn = partial(qa_colloator_fn, tokenizer=tokenizer,
                         img_transform=image_processor)
    dataloader = torch_data.DataLoader(
        qa_dataset, batch_size=1, collate_fn=collate_fn)

    keywords = ['###']
    ans_file = open(answers_file, "w")
    question_idx = 0

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, f'Generating answers'):
            num_beams = 3
            input_size = batch['input_ids'].shape[-1]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, tokenizer, input_size)
            # print(f'Input: {tokenizer.batch_decode(batch["input_ids"])}'
            #       f'input_ids: {batch["input_ids"]}'
            #       f'attn_mask: {batch["attention_mask"]}')

            output = model.generate(
                input_ids=batch['input_ids'].cuda(),
                images=batch['images'].half().cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                temperature=0.7,
                max_new_tokens=1024,
                num_beams=num_beams,
                # do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria],
                repetition_penalty=1.1)

            for question, output_ids in zip(batch['raw_questions'], output.sequences):
                response = tokenizer.decode(
                    output_ids[input_size:], skip_special_tokens=True)
                if response.count('###'):
                    response = response[: response.index('###')]
                response = response.strip()
                # print(f'{question}, {response}\n')

                ans_file.write(json.dumps({
                    "question_id": question_idx,
                    "prompt": question,
                    "text": response,
                    "model_id": args.model_name
                }) + "\n")
                ans_file.flush()
                question_idx += 1
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="default")
    args = parser.parse_args()

    eval_model(args)
