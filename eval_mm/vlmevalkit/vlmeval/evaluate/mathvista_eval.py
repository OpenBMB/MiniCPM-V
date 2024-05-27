from vlmeval.evaluate.misc import build_judge
from vlmeval.smp import *
from vlmeval.utils import track_progress_rich
from vlmeval.utils.matching_util import can_infer


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathvista_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        if line['question_type'] == 'multi_choice':
            ans = line['answer_option']
            choices = list_to_dict(eval(line['choices']))
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            if line['answer_type'] == 'integer':
                res = int(response)
                ans = int(line['answer'])
            elif line['answer_type'] == 'float':
                res = float(response)
                ans = float(line['answer'])
            else:
                res = str(res)
                ans = str(ans)
    except ValueError:
        pass

    if res == ans:
        return res if prefetch else True
    else:
        return False


def MathVista_auxeval(model, line):
    prompt = build_mathvista_gpt4_prompt(line)
    log = ''
    retry = 5
    if post_check(line, prefetch=True):
        res = post_check(line, prefetch=True)
        return dict(log='Prefetch succeed', res=res)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)
        if res is None:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def MathVista_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    skill_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['task']
        tot['Overall'] += 1
        try:
            skills = eval(item['skills'])
        except SyntaxError:
            skills = [item['skills']]
        for skill in skills:
            if skill not in skill_list:
                skill_list.append(skill)
            tot[skill] += 1
        tot[cate] += 1
        if item['log'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
            for skill in skills:
                fetch[skill] += 1
        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1
            for skill in skills:
                hit[skill] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Task&Skill'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res)
    return res


def MathVista_eval(eval_file, **judge_kwargs):
    logger = get_logger('Evaluation')
    model = judge_kwargs['model']

    suffix = eval_file.split('.')[-1]
    storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
    tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
    nproc = judge_kwargs.pop('nproc', 4)

    if osp.exists(storage):
        logger.warning(f'GPT scoring file {storage} already exists, will reuse it in MathVista_eval. ')
    else:
        data = load(eval_file)
        model = build_judge(max_tokens=128, **judge_kwargs)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        tups = [(model, line) for line in lines]
        indices = [line['index'] for line in lines]

        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)
        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]

        if len(indices):
            new_results = track_progress_rich(
                MathVista_auxeval, tups, nproc=nproc, chunksize=nproc,
                keys=indices, save=tmp_file)
            ans = load(tmp_file)
            for k, v in zip(indices, new_results):
                assert k in ans
                assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

        log_map, res_map = {}, {}
        all_inds = [line['index'] for line in lines]
        for k in all_inds:
            log_map[k] = ans[k]['log']
            res_map[k] = ans[k]['res']
        data['res'] = [res_map[idx] for idx in data['index']]
        data['log'] = [log_map[idx] for idx in data['index']]
        dump(data, storage)

    score = MathVista_acc(storage)
    score_pth = storage.replace('.xlsx', '_score.csv')

    dump(score, score_pth)
    logger.info(f'MathVista_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Score: ')
    logger.info(score)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM Answers. ')
    parser.add_argument('data', type=str, help='The question set for inference, in excel / tsv / json format. ')
    parser.add_argument(
        '--model',
        type=str,
        help='The LLM (GPT) used for inference. ',
        default='gpt-4-turbo',
        choices=['gpt-4-0613', 'gpt-4-turbo', 'chatgpt-1106', 'chatgpt-0613'])
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
    MathVista_eval(eval_file=args.data, **judge_kwargs)
