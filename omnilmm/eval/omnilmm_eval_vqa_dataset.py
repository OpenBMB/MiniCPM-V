import argparse
import itertools
import json
import os
import random
import pathlib
from PIL import Image
from functools import partial
from typing import Optional

import torch
import torch.utils.data as torch_data
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from omnilmm.eval.omnilmm_vqa import init_omnilmm, qa_colloator_fn, KeywordsStoppingCriteria, wrap_question_with_default_conv
from omnilmm.data.datasets import SingleDataSourceDataset
from omnilmm.data.data_processors import register_data_path, vqa_instruction_templates
from utils.vqa import VQA
from utils.vqa_eval import VQAEval

from utils.vqa_dataset import VQADataset
from utils.vqa_evaluate import evaluate_dataset

ds_collections = {
    'vqav2-val': {
        'question': '/data/public/multimodal/multimodal_data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': '/data/public/multimodal/multimodal_data/VQAv2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textVQA': {
        'metric': 'vqa_score',
    },
    'docVQA': {
        'metric': 'anls',
    },
    'ocrVQA': {
        'metric': 'has_answer',
    },
    'STVQA': {
        'metric': 'has_answer',
    },
    'pretrain_eval_eval': {
        'metric': 'acc'
    },
    'pretrain_eval_train': {
        'metric': 'acc'
    }
}


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class VQAEvalJSONDataset(torch_data.Dataset):
    def __init__(self, ds_path, question_process, max_size=-1, use_vqa_template=True) -> None:
        self.data = json.load(open(ds_path))['questions']

        self.use_vqa_template = use_vqa_template
        self.question_process = question_process

        if max_size == -1:
            max_size = len(self.data)
        self.max_size = max_size

    def __getitem__(self, index):
        item = self.data[index]

        image_id = int(item['image_id'])
        image_path = f'/data/public/multimodal/hanyifeng/vqav2_test2015/test2015/COCO_test2015_{image_id:012}.jpg'
        image = Image.open(image_path).convert('RGB')

        question = item['question']
        if self.use_vqa_template:
            question = vqa_instruction_templates(question, 3)

        question = self.question_process(question)

        return {
            'image': image,
            'question': question,
            'raw_question': item['question'],
            'question_id': item['question_id']
        }

    def __len__(self):
        return self.max_size


class VQAEvalDataset(torch_data.Dataset):
    def __init__(self, ds_name, question_process, max_size=-1, use_vqa_template=True) -> None:
        self.data = SingleDataSourceDataset(
            ds_name, *register_data_path[ds_name](), intent='eval')

        self.use_vqa_template = use_vqa_template
        self.question_process = question_process

        if max_size == -1:
            max_size = len(self.data)
        self.max_size = max_size

    def __getitem__(self, index):
        item = self.data[index]

        image = item['image']

        question = item['origin_question']
        if self.use_vqa_template:
            question = vqa_instruction_templates(question)

        question = self.question_process(question)

        return {
            'image': image,
            'question': question,
            'raw_question': item['origin_question'],
            'question_id': item['metainfo']['origin_idx']
        }

    def __len__(self):
        return self.max_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    # TextVQA, DocVQA, OCRVQA, STVQA
    parser.add_argument('--ds_name', type=str, default='TextVQA')
    parser.add_argument(
        "--load_tsv",
        action="store_true",
        default=False,
        help="whether load dataset from tsv"
    )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_sample', type=int, default=-1)
    parser.add_argument('--answer_file', type=str)
    parser.add_argument(
        "--load_json",
        action="store_true",
        default=False,
        help="whether load dataset from json"
    )
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    print(f'Init Rank-{torch.distributed.get_rank()}')
    if pathlib.Path(args.answer_file).exists():
        if torch.distributed.get_rank() == 0:
            print(f'{args.answer_file} Answer file exists, skip run.')
            ds_meta = ds_collections[args.ds_name]
            if args.ds_name == 'vqav2-val':
                question_ids = [x['question_id']
                                for x in json.load(open(args.answer_file))]
                vqa = VQA(ds_meta['annotation'], ds_meta['question'],
                          question_ids=set(question_ids))
                results = vqa.loadRes(
                    resFile=args.answer_file, quesFile=ds_meta['question'])
                vqa_scorer = VQAEval(vqa, results, n=2)
                vqa_scorer.evaluate()
                print(vqa_scorer.accuracy)
            else:
                acc = evaluate_dataset(
                    dataset_name=args.ds_name,
                    answer_file_path=args.answer_file,
                    model_name=args.checkpoint, method=ds_meta['metric'])
                print(f"{args.ds_name} : {acc} ")
        exit()

    model, image_processor, image_token_len, tokenizer = init_omnilmm(
        args.checkpoint)

    random.seed(args.seed)
    if args.load_tsv:
        dataset = VQAEvalDataset(args.ds_name, partial(
            wrap_question_with_default_conv, image_token_len=image_token_len), max_size=args.max_sample)
    elif args.load_json:
        dataset = VQAEvalJSONDataset(args.ds_name, partial(
            wrap_question_with_default_conv, image_token_len=image_token_len), max_size=args.max_sample)
    else:
        dataset = VQADataset(args.ds_name, partial(
            wrap_question_with_default_conv, image_token_len=image_token_len), max_size=args.max_sample)

    print(f'Dataset size is {len(dataset)}')

    collate_fn = partial(qa_colloator_fn, tokenizer=tokenizer,
                         img_transform=image_processor)
    dataloader = torch_data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    keywords = ['###']
    print(f'Dataloader size is {len(dataloader)}')

    outputs = []
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
                max_new_tokens=10,
                num_beams=num_beams,
                # do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria],
                repetition_penalty=1.1)

            # FIXME: simple fix missing key error for vqav2
            if 'gt_answers' not in batch:
                batch['gt_answers'] = batch['raw_questions'][:]

            for question, output_ids, question_id, gt_answers in zip(batch['raw_questions'], output.sequences, batch['question_id'], batch['gt_answers']):
                response = tokenizer.decode(
                    output_ids[input_size:], skip_special_tokens=True)
                if response.count('###'):
                    response = response[: response.index('###')]

                response = response.strip()
                if response.startswith('Assistant:'):
                    response = response[len('Assistant:'):].strip()

                # print(f'Q: {question_id} {question}, A: {response}', flush=True)

                outputs.append({
                    'question_id': question_id,
                    'raw_question': question,
                    'answer': response,
                    'gt_answers': gt_answers,
                })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    print(f'Merged outputs: {len(merged_outputs)}')
    question_ids = [x['question_id'] for x in merged_outputs]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.ds_name} ...", flush=True)
        answers_file_path = args.answer_file

        with open(answers_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_outputs, f, ensure_ascii=False)

        ds_meta = ds_collections[args.ds_name]
        if args.ds_name == 'vqav2-val':
            vqa = VQA(ds_meta['annotation'], ds_meta['question'],
                      question_ids=set(question_ids))
            results = vqa.loadRes(resFile=answers_file_path,
                                  quesFile=ds_meta['question'])
            vqa_scorer = VQAEval(vqa, results, n=2)
            vqa_scorer.evaluate()
            print(vqa_scorer.accuracy)
        else:
            acc = evaluate_dataset(
                dataset_name=args.ds_name,
                answer_file_path=answers_file_path,
                model_name=args.checkpoint, method=ds_meta['metric'])
            print(f"{args.ds_name} : {acc} ")

        # result = {}
        # result[args.ds_name] = acc
        # result_dir = os.path.dirname(answers_file_path)
        # result_path = os.path.join(result_dir, 'result.json')

        # with open(result_path, "a") as f:
        #     f.write("\n")
        #     f.write(json.dumps(result, indent=4))
    torch.distributed.barrier()
