import argparse
import numpy as np
import pandas as pd
import os.path as osp
from vlmeval.evaluate.misc import build_judge
from vlmeval.smp import *
from vlmeval.utils import track_progress_rich

rule_dict = {
    'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'}  # noqa: E501
}


def get_eval(judge, content):
    return judge.generate(content)


def parse_score(review):
    logger = get_logger('Evaluation')
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            logger.error('error', review)
            return [-1, -1]
    except Exception as e:
        logger.error(e, 'error', review)
        return [-1, -1]


def build_prompt(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[Context]\n{cap_str}\n\n'
               f'[Question]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
               f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
               f'[System]\n{prompt}\n\n')
    return content


def LLaVABench_atomeval(model, prompt):
    review = get_eval(model, prompt)
    scores = parse_score(review)
    return scores


def LLaVABench_score(data):
    cates = ['overall'] + list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['split'].append(c)
        sub = data[data['category'] == c] if c != 'overall' else data
        ret['Relative Score (main)'].append(np.mean(sub['score']) / np.mean(sub['gpt4_score']) * 100)
        ret['VLM Score'].append(np.mean(sub['score']) * 10)
        ret['GPT4 Score'].append(np.mean(sub['gpt4_score']) * 10)
    return pd.DataFrame(ret)


def LLaVABench_eval(eval_file, **judge_kwargs):
    suffix = '.' + eval_file.split('.')[-1]
    record_file = eval_file.replace(suffix, '_openai_result' + suffix)
    score_file = eval_file.replace(suffix, '_score.csv')
    nproc = judge_kwargs.pop('nproc', 4)

    if not osp.exists(record_file):
        data = load(eval_file)
        lines = [data.iloc[i] for i in range(len(data))]
        model = build_judge(
            temperature=0.2,
            system_prompt='You are a helpful and precise assistant for checking the quality of the answer.',
            **judge_kwargs)
        prompts = [build_prompt(line) for line in lines]
        tups = [(model, prompt) for prompt in prompts]
        scores = track_progress_rich(LLaVABench_atomeval, tups, nproc=nproc, chunksize=nproc)
        data['gpt4_score'] = [x[0] for x in scores]
        data['score'] = [x[1] for x in scores]
        dump(data, record_file)

    data = load(record_file)
    ret = LLaVABench_score(data).round(1)
    print(ret)
    dump(ret, score_file)
    return ret


def parse_args():
    parser = argparse.ArgumentParser(description='LLaVABench Evaluation. ')
    parser.add_argument('data', type=str, help='The question set for inference, in excel / tsv / json format. ')
    parser.add_argument(
        '--model', type=str, help='The LLM (GPT) used for inference. ', default='gpt-4-turbo',
        choices=['gpt-4-0613', 'gpt-4-turbo', 'chatgpt-1106', 'chatgpt-0613', 'gpt-4-0314'])
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    load_env()
    args = parse_args()
    judge_kwargs = dict(model=args.model, nproc=args.nproc, verbose=args.verbose)
    if 'OPENAI_API_KEY_JUDGE' in os.environ and os.environ['OPENAI_API_KEY_JUDGE']:
        judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
    if 'OPENAI_API_BASE_JUDGE' in os.environ and os.environ['OPENAI_API_BASE_JUDGE']:
        judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

    LLaVABench_eval(eval_file=args.data, **judge_kwargs)
