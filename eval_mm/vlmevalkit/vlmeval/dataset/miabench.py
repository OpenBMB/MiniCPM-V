import json
import os

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich


def generate_prompt(d):
    question = d['question']
    weights = eval(d['component_weight'])
    components = eval(d['components'])
    num_of_component = int(d['num_of_component'])
    response = d['prediction']

    if num_of_component == 1:
        components = f"The first component is: '{components[0]}'. "
        score = f"The first component is worth: {weights[0]} scores. "
    elif num_of_component == 2:
        components = f"The first component is: '{components[0]}', and the second component is '{components[1]}'. "
        score = f"The first and second component is each worth {weights[0]} and {weights[1]} scores. "
    elif num_of_component == 3:
        components = (
            f"The first component is: '{components[0]}', and the second component is '{components[1]}', "
            f"and the third component is '{components[2]}'. "
        )
        score = (
            "The first, second, and third component is each worth "
            f"{weights[0]}, {weights[1]}, and {weights[2]} scores."
        )
    elif num_of_component == 4:
        components = (
            f"The first component is: '{components[0]}', and the second component is '{components[1]}', "
            f"and the third component is '{components[2]}', and the fourth component is '{components[3]}'. "
        )
        score = (
            "The first, second, third, and fourth component is each worth "
            f"{weights[0]}, {weights[1]}, {weights[2]}, and {weights[3]} scores."
        )
    elif num_of_component == 5:
        components = (
            f"The first component is: '{components[0]}', and the second component is '{components[1]}', "
            f"and the third component is '{components[2]}', and the fourth component is '{components[3]}', "
            f"and the fifth component is '{components[4]}'. "
        )
        score = (
            "The first, second, third, fourth, and fifth component is each worth "
            f"{weights[0]}, {weights[1]}, {weights[2]}, {weights[3]}, and {weights[4]} scores."
        )

    return (
        "Here is an instruction for a multimodal LLM: '"
        f"{question}"
        "'. You need to grade if the response from the model follows each component of the instruction. "
        f"{components}"
        "The response is: '"
        f"{response}"
        "'. You need to score the response and be strict. The total score ranges from 0 to 10, "
        "depending on if the response follows the instruction. "
        f"{score}"
        "List scores of each component, and the total score in one sentence in this format: "
        "score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons."
    )


def process_rawscore(component_type, raw_score):
    first_sentence = raw_score.split('.')[0].split(',')
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(':')[1][1:].split('/')
        score = int(score_[0]) / int(score_[1])
        score_dict[component_type[i]] = score
    total_score_ = first_sentence[i + 1].split(':')[1][1:].split('/')
    total_score = int(total_score_[0]) / int(total_score_[1])
    score_dict['total_score'] = total_score
    return score_dict


def get_score_dict(data, score_raw):
    cat_score_dict = {}
    for i in range(len(data)):
        try:
            cmp = data['component_type'][i][2:-2]
            cmp_list = cmp.split('\', \'')
            score_dict = process_rawscore(cmp_list, score_raw[i])
            for key, val in score_dict.items():
                if key not in cat_score_dict.keys():
                    cat_score_dict[key] = [val]
                else:
                    cat_score_dict[key].append(val)
        except:
            pass
    cat_score_dict_average = {}
    for key, val in cat_score_dict.items():
        cat_score_dict_average[key] = sum(val) / len(val)
    return cat_score_dict_average


class MIABench(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'MIA-Bench': 'https://opencompass.openxlab.space/utils/VLMEval/Mia-Bench.tsv',
    }
    DATASET_MD5 = {
        'MIA-Bench': '0b9de595f4dd40af18a69b94d89aba82',
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        judge_name = judge_kwargs.pop('model', 'gpt-4o')

        model = build_judge(model=judge_name, **judge_kwargs)
        suffix = eval_file.split('.')[-1]

        storage = eval_file.replace(f'.{suffix}', f'_{judge_name}.xlsx')  # noqa: F841
        tmp_file = eval_file.replace(f'.{suffix}', f'_{judge_name}.pkl')  # noqa: F841
        nproc = judge_kwargs.pop('nproc', 4)  # noqa: F841

        if not osp.exists(storage):
            data = load(eval_file)
            num_samples = len(data)
            lines = [data.loc[i] for i in range(num_samples)]
            prompts = [generate_prompt(line) for line in lines]
            org_data = MIABench('MIA-Bench').data
            img_map = {x: y for x, y in zip(org_data['index'], org_data['image'])}
            image_b64 = [img_map[idx] for idx in data['index']]
            indices = list(data['index'])
            mm_messages = [
                dict(message=[
                    dict(type='text', value=prompt),
                    dict(type='image', value=f'data:image/jpeg;base64,{b64}')
                ])
                for prompt, b64 in zip(prompts, image_b64)
            ]

            res = {}
            if osp.exists(tmp_file):
                res = load(tmp_file)

            jobs = {k: v for k, v in zip(indices, mm_messages) if k not in res}
            job_keys = list(jobs.keys())
            job_vals = [jobs[k] for k in job_keys]

            resps = track_progress_rich(
                model.generate,
                job_vals,
                nproc=nproc,
                chunksize=nproc,
                keys=job_keys,
                save=tmp_file,
            )
            for k, resp in zip(job_keys, resps):
                res[k] = resp
            data['score_raw'] = [res[idx] for idx in indices]
            dump(data, storage)

        goresult = load(storage)
        results = get_score_dict(goresult, goresult['score_raw'])
        result_pth = storage.replace('.xlsx', '_score.csv')
        results_pd = pd.DataFrame.from_dict(list(results.items()))
        dump(results_pd, result_pth)

        return results
