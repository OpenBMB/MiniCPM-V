import io
import os
import json
import torch
import base64
import argparse

import torch.utils.data as torch_data

import tqdm
from functools import partial
from PIL import Image

from omnilmm.conversation import conv_templates
from omnilmm.eval.muffin_vqa import torch_pad_sequence
from omnilmm.eval.zephyr_mm_chat import init_zephyr_mm, wrap_question_for_zephyr_mm
from omnilmm.eval.muffin_vqa import expand_question_into_multimodal


class MultimodalQADataset(torch_data.Dataset):
    def __init__(self, qa_file, question_process, max_len=-1):
        '''
        qa_file: jsonl file that each line is a dict like {
            'image': b64img,
            'question': question_text
        }
        '''
        super().__init__()

        self.qa_file = qa_file
        try:
            self.qa_data = [json.loads(line) for line in open(self.qa_file)]
            if isinstance(self.qa_data[0], list):
                # unwrap one-line json question file
                self.qa_data = self.qa_data[0]
        except:
            try:
                with open(self.qa_file, "r") as f:
                    self.qa_data = json.load(f)
            except:
                raise ValueError("Wrong input data format!")

        self.question_process = question_process
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.qa_data[index]
        if "image_id" in item.keys():
            imgid = item["image_id"]

        img_b64 = item['image']
        if len(img_b64) > 100:
            image = Image.open(io.BytesIO(
                base64.b64decode(img_b64))).convert('RGB')
        else:
            image = Image.open(img_b64).convert('RGB')

        metainfo = {key: value for key, value in item.items() if key not in [
            "image_id", "question", "image"]}

        raw_question = item['question']
        if self.question_process != None:
            question_text = self.question_process(raw_question)
        else:
            question_text = raw_question
        return {
            'image': image,
            'raw_question': raw_question,
            'question': question_text,
            'image_id': imgid,
            'metainfo': metainfo,
        }

    def __len__(self):
        if self.max_len != -1:
            return self.max_len
        return len(self.qa_data)


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def wrap_question_with_default_conv(question_text, image_token_len, coh_prefix_id=None):
    question_text = expand_question_into_multimodal(
        question_text, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)
    conv = conv_templates['default'].copy()
    conv.messages = []
    conv.sep = '\n###'

    conv.append_message(conv.roles[0], question_text)
    prompt = conv.get_prompt()
    if coh_prefix_id != None:
        prompt += f'Assistant: {coh_prefix[coh_prefix_id]}'
    return prompt


def qa_colloator_fn(data_list, tokenizer, img_transform):
    questions = [x['question'] for x in data_list]
    # print(questions)
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
    if 'image_id' in data_list[0]:
        data['image_id'] = [x['image_id'] for x in data_list]
    if 'metainfo' in data_list[0]:
        data['metainfo'] = [x['metainfo'] for x in data_list]

    return data


def eval_model(args):
    model, image_processor, image_token_len, tokenizer = init_zephyr_mm(
        args.model_name, tune_clip=args.tune_clip)

    answer_dir = '/'.join(args.answers_file.split("/")[:-1])
    print(answer_dir)
    assert os.path.exists(answer_dir), f'Expecting {answer_dir} to be existing'
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    qa_dataset = MultimodalQADataset(args.question_file, partial(
        wrap_question_with_default_conv, image_token_len=image_token_len, coh_prefix_id=args.coh_prefix_id),
        max_len=300)

    collate_fn = partial(qa_colloator_fn, tokenizer=tokenizer,
                         img_transform=image_processor)
    dataloader = torch_data.DataLoader(
        qa_dataset, batch_size=1, collate_fn=collate_fn)

    ans_file = open(answers_file, "w")
    question_idx = 0

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, f'Generating answers'):
            num_beams = 3
            input_size = batch['input_ids'].shape[-1]
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
                repetition_penalty=1.1)

            for question, image_id, metainfo, output_ids in zip(batch['raw_questions'], batch['image_id'], batch['metainfo'], output.sequences):
                response = tokenizer.decode(
                    output_ids[input_size:], skip_special_tokens=True)
                if response.count('###'):
                    response = response[: response.index('###')]
                response = response.strip()
                # print(f'{question}, {response}\n')

                ans_file.write(json.dumps({
                    "question_id": question_idx,
                    "image_id": image_id,
                    "prompt": question,
                    "text": response,
                    "metainfo": metainfo,
                    "model_id": args.model_name,
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
    parser.add_argument("--coh-prefix-id", type=int, default=None)
    args = parser.parse_args()

    eval_model(args)
