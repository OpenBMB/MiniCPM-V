import argparse
import itertools
import json
import glob
import os
import io
import pandas
import random
import pathlib
from PIL import Image
from functools import partial
from typing import Optional
from collections import defaultdict

import torch
import torch.utils.data as torch_data
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from omnilmm.eval.omni_lmm_chat import init_omni_lmm, wrap_question_for_omni_lmm
from omnilmm.data.datasets import SingleDataSourceDataset
from omnilmm.data.data_processors import register_data_path, vqa_instruction_templates
from utils.vqa import VQA
from utils.vqa_eval import VQAEval
from omnilmm.eval.omnilmm_eval_vqa_dataset import InferenceSampler, ds_collections, VQAEvalJSONDataset, VQAEvalDataset
from omnilmm.eval.omnilmm_vqa import torch_pad_sequence
from utils.vqa_dataset import VQADataset
from utils.vqa_evaluate import evaluate_dataset


def judge_multichoice(answer: str, gt: str):
    answer = answer.lower().strip()
    gt = gt.lower().strip()
    assert len(gt) == 1
    
    if answer == gt:
        return True
    if answer[0] == gt:
        return True
    if answer.startswith(f'the answer is {gt}'):
        return True
    if answer.startswith(f'the correct answer is {gt}'):
        return True
    if answer.startswith(f'answer: {gt}'):
        return True
    if answer.startswith(f'answer:\n{gt}'):
        return True

    if answer.startswith(f'the answer is ({gt})'):
        return True
    if answer.startswith(f'the correct answer is ({gt})'):
        return True
    if answer.startswith(f'answer: ({gt})'):
        return True
    if answer.startswith(f'answer:\n({gt})'):
        return True

    return False

class PretrainEvalDataset(torch_data.Dataset):
    def __init__(self, ds_name, question_process, max_size=-1, rand_sample=True) -> None:
        self.data = SingleDataSourceDataset(
            ds_name, *register_data_path[ds_name](), intent='eval')
        print(
            f'Load pretrain-eval data from {ds_name}, origin size is {len(self.data)}.')
        self.question_process = question_process

        if max_size == -1:
            max_size = len(self.data)
        self.max_size = max_size

        if rand_sample:
            self.line_numbers = random.sample(
                list(range(len(self.data))), self.max_size)
            random.shuffle(self.line_numbers)
        else:
            self.line_numbers = list(range(len(max_size)))

    def __getitem__(self, index):
        line_number = self.line_numbers[index]
        item = self.data[line_number]

        image = item['image']
        raw_question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        question_input_ids = self.question_process(raw_question)['input_ids']

        return {
            'image': image,
            'question_input_ids': question_input_ids,
            'answer': answer,
            'raw_question': raw_question,
            'question_id': item['metainfo']['origin_idx'],
            'origin_dataset': item['metainfo']['origin_dataset'],
        }

    def __len__(self):
        return self.max_size


def construct_option_str(options):
    out = ''
    code = ord('A')
    for option in options:
        out += f'\n{chr(code)}. {option}'
        code += 1
    return out


class MMMUDevDataset(torch_data.Dataset):
    def __init__(self, ds_name, question_process, max_size=-1, rand_sample=True) -> None:
        assert ds_name == 'MMMUDev'
        self.data_files = list(
            glob.glob('/home/yutianyu/MMMU_dev/*/validation*'))
        data_chuncks = [list(pandas.read_parquet(file).iterrows())
                        for file in self.data_files]
        self.data = sum(data_chuncks, [])
        self.data = [x for x in self.data if x[1]['question_type'] ==
                     'multiple-choice' and x[1]['answer'] in 'ABCDEFGHIJKLMN']

        self.question_process = question_process
        if max_size == -1:
            max_size = len(self.data)
        self.max_size = max_size

        if rand_sample:
            self.line_numbers = random.sample(
                list(range(len(self.data))), self.max_size)
            random.shuffle(self.line_numbers)
        else:
            self.line_numbers = list(range(len(max_size)))

    def __getitem__(self, index):
        line_number = self.line_numbers[index]
        item = self.data[line_number][1]

        question = item['question']
        options = construct_option_str(eval(item['options']))
        answer = item['answer']
        raw_question = f'{question}\nOptions:{options}'
        raw_question.replace('<image 1>', '<image>')

        image = Image.open(io.BytesIO(item['image_1']['bytes'])).convert('RGB')
        question_input_ids = self.question_process(raw_question)['input_ids']

        return {
            'image': image,
            'question_input_ids': question_input_ids,
            'answer': answer,
            'raw_question': raw_question,
            'origin_dataset': item['subfield'],
            'question_id': item['id']
        }

    def __len__(self):
        return len(self.data)


class MMBDataset(torch_data.Dataset):
    def __init__(self, ds_name, question_process, max_size=-1, rand_sample=True) -> None:
        self.data = SingleDataSourceDataset(
            ds_name, *register_data_path[ds_name](), intent='eval')
        print(
            f'Load pretrain-eval data from {ds_name}, origin size is {len(self.data)}.')
        self.question_process = question_process

        if max_size == -1:
            max_size = len(self.data)
        self.max_size = max_size

        if rand_sample:
            self.line_numbers = random.sample(
                list(range(len(self.data))), self.max_size)
            random.shuffle(self.line_numbers)
        else:
            self.line_numbers = list(range(len(max_size)))

    def __getitem__(self, index):
        line_number = self.line_numbers[index]
        item = self.data[line_number]

        image = item['image']
        raw_question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        question_input_ids = self.question_process(raw_question)['input_ids']

        question_id_str = eval(item['metainfo']['origin_split'])[2]

        return {
            'image': image,
            'question_input_ids': question_input_ids,
            'answer': answer,
            'raw_question': raw_question,
            'question_id': question_id_str,
            'origin_dataset': item['metainfo']['origin_dataset']
        }

    def __len__(self):
        return self.max_size


def omni_qa_colloator_fn(data_list, tokenizer, img_transform):
    input_ids = [torch.as_tensor(x['question_input_ids']) for x in data_list]
    attn_mask = [torch.as_tensor([1] * len(x)) for x in input_ids]

    input_ids = torch_pad_sequence(
        input_ids, tokenizer.pad_token_id, padding_side='left')
    attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')

    images = [img_transform(x['image']) for x in data_list]
    images = torch.stack(images)

    raw_questions = [x['raw_question'] for x in data_list]
    data = {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'raw_questions': raw_questions,
    }

    if 'question_id' in data_list[0]:
        data['question_id'] = [x['question_id'] for x in data_list]
    if 'origin_dataset' in data_list[0]:
        data['origin_dataset'] = [x['origin_dataset'] for x in data_list]
    if 'answer' in data_list[0]:
        data['gt_answers'] = [x['answer'] for x in data_list]
    if 'image_id' in data_list[0]:
        data['image_id'] = [x['image_id'] for x in data_list]
    if 'metainfo' in data_list[0]:
        data['metainfo'] = [x['metainfo'] for x in data_list]

    return data


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
    parser.add_argument('--max_tokens', type=int, default=10)
    parser.add_argument('--answer_file', type=str)
    parser.add_argument(
        "--load_json",
        action="store_true",
        default=False,
        help="whether load dataset from json"
    )
    parser.add_argument('--tune_clip', type=str,
                        default='True', choices=['True', 'False'])
    args = parser.parse_args()

    args.tune_clip = True if args.tune_clip == 'True' else False

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    print(f'Init Rank-{torch.distributed.get_rank()}')
    model, image_processor, image_token_len, tokenizer = init_omni_lmm(
        args.checkpoint, tune_clip=args.tune_clip)
    random.seed(args.seed)

    if args.ds_name.startswith('pretrain_eval'):
        dataset = PretrainEvalDataset(args.ds_name, partial(
            wrap_question_for_omni_lmm, image_token_len=image_token_len, tokenizer=tokenizer), max_size=args.max_sample)
    elif args.ds_name.startswith('eval_mmbench'):
        dataset = MMBDataset(args.ds_name, partial(
            wrap_question_for_omni_lmm, image_token_len=image_token_len, tokenizer=tokenizer), max_size=args.max_sample)
    elif args.ds_name.startswith('MMMUDev'):
        dataset = MMMUDevDataset(args.ds_name, partial(
            wrap_question_for_omni_lmm, image_token_len=image_token_len, tokenizer=tokenizer), max_size=args.max_sample)
    else:
        raise NotImplementedError

    print(f'Dataset size is {len(dataset)}')

    collate_fn = partial(omni_qa_colloator_fn, tokenizer=tokenizer,
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
    print(f'Dataloader size is {len(dataloader)}')

    outputs = []
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
                # temperature=0.7,
                max_new_tokens=args.max_tokens,
                num_beams=num_beams,
                # do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1)

            # FIXME: simple fix missing key error for vqav2
            if 'gt_answers' not in batch:
                batch['gt_answers'] = batch['raw_questions'][:]
            # print(output.scores, flush=True)
            for question, output_ids, question_id, gt_answers, origin_dataset in zip(batch['raw_questions'], output.sequences, batch['question_id'], batch['gt_answers'], batch['origin_dataset']):
                response = tokenizer.decode(
                    output_ids[input_size:], skip_special_tokens=True)
                response = response.strip()

                # print(f'Q: {question_id} {question}, A: {response}, GT {gt_answers}', flush=True)

                outputs.append({
                    'question_id': question_id,
                    'raw_question': question,
                    'answer': response,
                    'gt_answers': gt_answers,
                    'origin_dataset': origin_dataset
                })
                if isinstance(dataset, MMBDataset):
                    if len(question_id.split('-')) == 3:
                        _, category, qid = question_id.split('-')
                    else:
                        _, category, _, qid = question_id.split('-')
                    outputs[-1]['origin_dataset'] = category
                    outputs[-1]['question_id'] = qid

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

        ds_meta = ds_collections.get(args.ds_name, {})
        if args.ds_name == 'vqav2-val':
            vqa = VQA(ds_meta['annotation'], ds_meta['question'],
                      question_ids=set(question_ids))
            results = vqa.loadRes(resFile=answers_file_path,
                                  quesFile=ds_meta['question'])
            vqa_scorer = VQAEval(vqa, results, n=2)
            vqa_scorer.evaluate()
            print(vqa_scorer.accuracy)
        elif args.ds_name.startswith('pretrain_eval'):
            n_correct = defaultdict(int)
            n_total = defaultdict(int)
            for item in merged_outputs:
                category = item['origin_dataset']
                qid = item['question_id']
                answer = item['answer']
                gt = item['gt_answers']
                n_total[category] += 1
                if judge_multichoice(answer, gt):
                    n_correct[category] += 1
            metrics = {}
            for category in n_total:
                metrics[category] = {
                    'acc': n_correct[category] / n_total[category],
                    'total': n_total[category],
                    'correct': n_correct[category]
                }
            print(metrics, flush=True)
        elif args.ds_name.startswith('MMMU'):
            n_correct = defaultdict(int)
            n_total = defaultdict(int)
            sum_total = 0
            sum_correct = 0
            for item in merged_outputs:
                category = item['origin_dataset']
                qid = item['question_id']
                answer = item['answer']
                gt = item['gt_answers']
                n_total[category] += 1
                sum_total += 1
                if judge_multichoice(answer, gt):
                    n_correct[category] += 1
                    sum_correct += 1
            metrics = {}
            for category in n_total:
                metrics[category] = {
                    'acc': n_correct[category] / n_total[category],
                    'total': n_total[category],
                    'correct': n_correct[category]
                }
            metrics['total'] = {'acc': sum_correct / sum_total,
                                'total': sum_total, 'correct': sum_correct}
            print(metrics, flush=True)
        elif args.ds_name.startswith('eval_mmbench'):
            qid2items = defaultdict(list)
            category2qids = defaultdict(set)
            for item in merged_outputs:
                category = item['origin_dataset']
                qid = item['question_id']
                category2qids[category].add(qid)
                qid2items[qid].append(item)

            n_correct = defaultdict(int)
            n_total = defaultdict(int)
            for category in category2qids:
                for qid in category2qids[category]:
                    all_correct = True
                    for item in qid2items[qid]:
                        answer = item['answer']
                        gt = item['gt_answers']
                        if not judge_multichoice(answer, gt):
                            all_correct = False
                            break
                    if all_correct:
                        n_correct[category] += 1
                    n_total[category] += 1
            sum_total = sum(n_total.values())
            sum_correct = sum(n_correct.values())
            metrics = {}
            for category in n_total:
                metrics[category] = {
                    'acc': n_correct[category] / n_total[category],
                    'total': n_total[category],
                    'correct': n_correct[category]
                }
            metrics['total'] = {'acc': sum_correct / sum_total,
                                'total': sum_total, 'correct': sum_correct}
            print(metrics, flush=True)
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
