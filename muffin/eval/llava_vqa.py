import os
import json
import torch
import argparse

import torch.utils.data as torch_data

import tqdm
from functools import partial
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor

from muffin import LlavaLlamaForCausalLM
from muffin.utils import disable_torch_init
from muffin.model.utils import build_transform
from muffin.data.datasets import MultimodalQADataset

from muffin.eval.muffin_vqa import patch_config, wrap_question_with_default_conv, qa_colloator_fn, KeywordsStoppingCriteria


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def init_llava(model_path, device=None):
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_config(model_name)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, tune_clip=False).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(
        '/data/public/multimodal/multimodal_data/models/openai_clip_large_img_encoder', torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    if False:
        vision_tower = model.model.vision_tower
    else:
        vision_tower = model.model.vision_tower[0]
    vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size //
                       vision_config.patch_size) ** 2

    def img_process_func(img):
        return image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
    print(image_token_len)
    return model, img_process_func, image_token_len, tokenizer


def eval_model(args):
    model, image_processor, image_token_len, tokenizer = init_llava(
        args.model_name)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    qa_dataset = MultimodalQADataset(args.question_file, partial(
        wrap_question_with_default_conv, image_token_len=image_token_len))

    collate_fn = partial(qa_colloator_fn, tokenizer=tokenizer,
                         img_transform=image_processor)
    # FIXME: use batch_size >= 1 result errors
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
                # num_beams=num_beams,
                do_sample=True,
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
