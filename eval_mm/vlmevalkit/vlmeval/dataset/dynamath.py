import re
import json
import sympy as sp
import numpy as np
import pandas as pd
from sympy import simplify, Eq, sympify, Pow, pi
from sympy.parsing.latex import parse_latex
import sys
import math
import os
import os.path as osp
import argparse

from .image_base import ImageBaseDataset
from .utils import build_judge
from ..utils import track_progress_rich
from ..smp import load, dump, d2df, toliststr


def preprocess(str1):
    if 0 <= str1.find("{") < str1.rfind("}"):
        str1 = str1[str1.find("{"): str1.rfind("}") + 1]
    str2 = str1.replace("\\", "")
    str2 = str2.replace("\\n", "\n")
    return str2


def transfer(str1):
    if "\u03c0" in str1:
        strs = str1.split('\u03c0')
        str1 = strs[0]
        return float(str1) * np.pi
    else:
        return float(str1)


def parse_answer(answer, answer_type="multiple choice"):
    if answer_type == "float":
        if answer.isdigit():
            return True, float(answer)
        else:
            parts = answer.split(' ')
            answer = parts[0]
            try:
                answer = transfer(answer)
                return True, answer
            except:
                return False, None
    elif answer_type == "multiple choice":
        if len(answer) == 1:
            return True, answer.upper()
        else:
            in_flag = [ch in answer.upper() for ch in 'ABCDE']
            if sum(in_flag) == 1:
                for ch in 'ABCDE':
                    if ch in answer.upper():
                        return True, ch
            return False, None
    else:
        return True, answer


def DynaMath_auxeval(model, line):
    pred = line['prediction']
    pred = preprocess(pred)

    succeed, short_answer = None, None
    try:
        dj = json.loads(pred, strict=False)
        short_answer = dj.get("short answer")
        assert short_answer is not None
        succeed, short_answer = parse_answer(short_answer, answer_type=line['anwser_type'])
        assert succeed
    except:
        # Failed to parse the JSON, use an auxiliary LLM to get the short answer
        if line['answer_type'] == 'multiple choice':
            inst = "Output the corresponing choice option, such as 'A', 'B', 'C', 'D', in a single line."
        elif line['answer_type'] == 'float':
            inst = "Output a three-digit floating-point number in a single line."
        else:
            inst = (
                "Output a short answer in a single line. Any float numbers in the answer "
                "should be formatted as three-digit floating-point numbers."
            )

        prompt = f"Free-form answer: {pred}\nInstruction: {inst}"
        response = pred
        succeed, short_answer = parse_answer(response, line['answer_type'])
        if not succeed:
            response = model.generate(prompt)
            succeed, short_answer = parse_answer(response, line['answer_type'])

    if line['answer_type'] == 'float':
        if succeed:
            diff = float(short_answer) - float(line['answer'])
            if abs(diff) <= 0.001:
                return dict(parse=True, extracted=short_answer, correct=True)
            else:
                return dict(parse=True, extracted=short_answer, correct=False)
        else:
            return dict(parse=False, extracted=None, correct=False)
    elif line['answer_type'] == 'multiple choice':
        if succeed:
            return dict(parse=True, extracted=short_answer, correct=(short_answer == line['answer']))
        else:
            if line['answer'] in pred[:3].upper():
                return dict(parse=False, extracted=None, correct=True)
            else:
                return dict(parse=False, extracted=None, correct=False)
    else:
        if succeed:
            return dict(parse=True, extracted=short_answer, correct=(short_answer.lower() in line['answer'].lower()))
        else:
            return dict(parse=False, extracted=None, correct=(short_answer.lower() in line['answer'].lower()))


class Dynamath(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {'DynaMath': 'https://opencompass.openxlab.space/utils/VLMEval/DynaMath.tsv'}
    DATASET_MD5 = {'DynaMath': 'b8425ad9a7114571fc9366e013699494'}
    GUIDE = """
## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere \
to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain \
detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:

"""
    EXAMPLE = {
        "solution": "[Detailed step-by-step explanation]",
        "short answer": "[Concise Answer]"
    }
    TEXT_EXAMPLE = json.dumps(EXAMPLE, indent=4)

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        prompt = f"## Question\n {line['question']}"
        if line['answer_type'] == 'multiple choice':
            inst = "Provide the corresponing choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif line['answer_type'] == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        prompt = prompt + self.GUIDE.format(INST=inst) + self.TEXT_EXAMPLE

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        judge_name = judge_kwargs.pop('model', 'gpt-4o-mini')

        model = build_judge(model=judge_name, **judge_kwargs)
        suffix = eval_file.split('.')[-1]

        storage = eval_file.replace(f'.{suffix}', f'_{judge_name}.xlsx')  # noqa: F841
        score_file = eval_file.replace(f'.{suffix}', f'_{judge_name}_score.csv')  # noqa: F841
        tmp_file = eval_file.replace(f'.{suffix}', f'_{judge_name}.pkl')  # noqa: F841
        nproc = judge_kwargs.pop('nproc', 6)  # noqa: F841

        res = load(tmp_file) if os.path.exists(tmp_file) else {}
        res = {k: v for k, v in res.items() if v is not None}

        model.system_prompt = """\
You are a helpful assistant that helps me to format free-form answers into a short answer according to the instruction.
"""
        if not osp.exists(storage):
            data = load(eval_file)
            lt = len(data)
            payloads = [dict(model=model, line=data.iloc[i]) for i in range(lt) if data.iloc[i]['index'] not in res]
            keys = [idx for idx in data['index'] if idx not in res]

            if len(keys):
                results = track_progress_rich(DynaMath_auxeval, payloads, nproc=nproc, save=tmp_file, keys=keys)
                for k, r in zip(keys, results):
                    res[k] = r

            data['parse'] = [res[idx]['parse'] for idx in data['index']]
            data['extracted'] = [res[idx]['extracted'] for idx in data['index']]
            data['correct'] = [res[idx]['correct'] for idx in data['index']]
            dump(data, storage)

        data = load(storage)
        # Calculate Average Accuracy
        score_avg = {}
        score_avg['Overall'] = np.mean(data['correct'])

        subs = set(data['subject'])
        for sub in subs:
            data_sub = data[data['subject'] == sub]
            score_avg[f'Subject-{sub}'] = np.mean(data_sub['correct'])

        lvls = set(data['knowledge_level'])
        for lvl in lvls:
            data_lvl = data[data['knowledge_level'] == lvl]
            score_avg[f'Level-{lvl}'] = np.mean(data_lvl['correct'])

        # Calculate the Worst Case Accuracy
        score_worst = {}
        data_worst = data[data['varid'] == 1]
        qid2corr = {idx: True for idx in data_worst['index']}
        lt = len(data)
        for i in range(lt):
            item = data.iloc[i]
            qid2corr[item['qid']] *= item['correct']
        data_worst['correct'] = [qid2corr[idx] for idx in data_worst['qid']]
        score_worst['Overall'] = np.mean(data_worst['correct'])

        subs = set(data_worst['subject'])
        for sub in subs:
            data_sub = data_worst[data_worst['subject'] == sub]
            score_worst[f'Subject-{sub}'] = np.mean(data_sub['correct'])

        lvls = set(data_worst['knowledge_level'])
        for lvl in lvls:
            data_lvl = data_worst[data_worst['knowledge_level'] == lvl]
            score_worst[f'Level-{lvl}'] = np.mean(data_lvl['correct'])

        d1 = {'Setting': 'Average'}
        d1.update(score_avg)
        d2 = {'Setting': 'Worst Case'}
        d2.update(score_worst)
        score = pd.concat([d2df(d1), d2df(d2)], ignore_index=True)

        dump(score, score_file)
        return score
