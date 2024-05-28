import os.path as osp
import pandas as pd
from tqdm import tqdm
from vlmeval.evaluate.misc import build_judge
from vlmeval.utils import can_infer, track_progress_rich, TSVDataset
from vlmeval.smp import *
import numpy as np

INTERNAL = os.environ.get('INTERNAL', 0)

abbrs = {
    'coarse_perception': 'CP',
    'finegrained_perception (instance-level)': 'FP-S',
    'finegrained_perception (cross-instance)': 'FP-C',
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}


def MMMU_preproc(data):
    logger = get_logger('Evaluation')
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    logger.info(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones. ')
    data['A'] = As
    data['B'] = Bs
    return data


def report_acc(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'l2-category', 'category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = abbrs[ab] if ab in abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
    return pd.DataFrame(res)


def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


def build_prompt_cn(question, options, prediction):
    tmpl = (
        '你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。'
        '你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。'
        '如果所有选项的意义都与答案显著不同，则输出 Z。'
        '你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。'
        '例 1:'
        '问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n'
        '例 2: \n'
        '问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: Z\n'
        '例 3: \n'
        '问题: {}?\n选项: {}\n答案: {}\n输出: '
    )
    return tmpl.format(question, options, prediction)


def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def prefetch_answer(item):
    choices = build_choices(item)
    return can_infer(item['prediction'], choices)


def extract_answer_from_item(model, item):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 3

    ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])

    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
        else:
            ret = can_infer(ans, choices)
            if ret:
                return dict(opt=ret, log=ans)
            else:
                logger.warning(f'Output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        retry -= 1

        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else []
            return dict(opt=rd.choice(options), log='Failed to predict, thus randomly generate one. ')


def prefetch_sub_data(sub_data, answer_map, verbose=False):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        idx = item['index']
        GT.append(answer_map[idx])
        PRED.append(prefetch_answer(item))
        if PRED[-1] and (GT[-1] != PRED[-1]):
            log = (
                f'Failed in Prefetching Rolling {i}: Answer is {GT[-1]}, '
                f"Prediction is {item['prediction']}, Pre-fetched is {PRED[-1]}. "
            )
            return dict(hit=0, log=log)
    flag = True
    for g, p in zip(GT, PRED):
        if g != p:
            flag = False
    ret = (dict(hit=1, log='Succeed During Pre-fetching'), ) if flag else (None, )
    ret = ret + (GT, PRED) if verbose else ret
    return ret if len(ret) > 1 else ret[0]


def eval_sub_data(model, sub_data, answer_map):
    res, GT, PRED = prefetch_sub_data(sub_data, answer_map, verbose=True)
    if res is not None:
        return res

    lt = len(sub_data)
    log = ''
    for i in range(lt):
        if PRED[i]:
            log += f'Rolling {i} Matched.\n'
        else:
            res = extract_answer_from_item(model, sub_data.iloc[i])
            opt, match_log = res['opt'], res['log']
            PRED[i] = opt
            if PRED[i] != GT[i]:
                log += (
                    f"Failed in Rolling {i}: Answer is {GT[i]}; Prediction is {sub_data.iloc[i]['prediction']}; "
                    f'Pre-fetched is {PRED[i]}; Match Log is {match_log}.\n'
                )
                return dict(hit=0, log=log)
            else:
                log += (
                    f"Rolling {i}: Answer is {GT[i]}, Prediction is {sub_data.iloc[i]['prediction']}, "
                    f'Pre-fetched is {PRED[i]}.\n'
                )

    return dict(hit=1, log=log)


def eval_data_groups(model, data_groups, answer_map, result, result_file, nproc=16):
    prefetched = [prefetch_sub_data(g, answer_map, verbose=False) for g in data_groups]
    remain = []
    for dg, pf in zip(data_groups, prefetched):
        if pf:
            result[dg.iloc[0]['index'] % 1e6] = pf
        else:
            remain.append(dg)
    dump(result, result_file)
    tups = [(model, x, answer_map) for x in remain]
    keys = [x.iloc[0]['index'] % 1e6 for x in remain]
    if len(tups) == 0:
        return

    if model is None:
        logger = get_logger('Evaluation')
        logger.warning('Exact Matching mode, will not do GPT-based answer matching. ')
        for k in keys:
            result[k] = dict(
                hit=0, log='Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.')
        dump(result, result_file)
        return

    res = track_progress_rich(
        eval_sub_data,
        tups,
        nproc=nproc,
        chunksize=nproc,
        save=result_file,
        keys=keys)
    result = load(result_file)
    for k, v in zip(keys, res):
        if k in result:
            assert result[k]['hit'] == v['hit'] and result[k]['log'] == v['log']
        else:
            result[k] = v
    dump(result, result_file)


def multiple_choice_eval(eval_file, dataset='default', **judge_kwargs):
    logger = get_logger('Evaluation')

    # assert dataset is not None
    dataset_map = {
        'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
    }
    if dataset in dataset_map:
        dataset = dataset_map[dataset]
    nproc = judge_kwargs.pop('nproc', 4)

    if listinstr(['mmbench', 'ccbench'], dataset.lower()):
        data = load(eval_file)
        data['index'] = [int(x) for x in data['index']]
        dump(data, eval_file)

    rd.seed(2680)
    suffix = eval_file.split('.')[-1]
    model = judge_kwargs['model']
    assert model in ['chatgpt-0613', 'exact_matching', 'gpt-4-0125']
    name_str_map = {
        'chatgpt-0613': 'openai',
        'gpt-4-0125': 'gpt4'
    }
    name_str = name_str_map[model] if model in name_str_map else model

    if model == 'exact_matching':
        model = None
    else:
        if INTERNAL or gpt_key_set():
            model = build_judge(**judge_kwargs)
        else:
            logger.error('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

    logger.info(f'Evaluating {eval_file}')
    result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    data = load(eval_file)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    if dataset != 'default':
        meta = TSVDataset(dataset).data
    else:
        logger.warning('Dataset is not provided, try to use the original `eval_file` as meta data. ')
        meta = load(eval_file)
        assert 'index' in meta and 'answer' in meta, 'Essentail columns missing in the eval_file.'

    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    cate_map = {i: c for i, c in zip(meta['index'], meta['category'])} if 'category' in meta else None
    l2_cate_map = {i: c for i, c in zip(meta['index'], meta['l2-category'])} if 'l2-category' in meta else None
    split_map = {i: c for i, c in zip(meta['index'], meta['split'])} if 'split' in meta else None

    if cate_map is not None and np.all([pd.isna(x) for x in cate_map.values()]):
        cate_map = None
    if l2_cate_map is not None and np.all([pd.isna(x) for x in l2_cate_map.values()]):
        l2_cate_map = None
    if split_map is not None and np.all([pd.isna(x) for x in split_map.values()]):
        split_map = None

    if listinstr(['MMMU'], dataset):
        data = MMMU_preproc(data)
        answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}

    data = data[data['index'].isin(answer_map)]
    data_main = data[data['index'] < int(1e6)]
    meta_idx_set = set(meta['index'])
    data_main = data_main[data_main['index'].isin(meta_idx_set)]

    lt = len(data_main)
    hit, tot = 0, 0

    data_groups = []
    for i in tqdm(range(lt)):
        # Dealing with the normal part
        item_main = data_main.iloc[i]
        idx = item_main['index']

        if idx in result:
            correct = result[idx]['hit']
            assert correct in [0, 1]
            hit += correct
            tot += 1
            continue

        sub_data = data[data['index'] % int(1e6) == idx]
        data_groups.append(sub_data)

    if len(data_groups):
        eval_data_groups(
            model=model,
            data_groups=data_groups,
            answer_map=answer_map,
            nproc=nproc,
            result=result,
            result_file=result_file)

    tmp_pth = f'/tmp/{timestr()}.xlsx'
    dump(data_main, tmp_pth)
    data_main = load(tmp_pth)

    res = load(result_file)
    indices = data_main['index']

    data_main['hit'] = [res[i]['hit'] for i in indices]
    data_main['log'] = [res[i]['log'] for i in indices]

    main_idx = data_main['index']
    if cate_map is not None:
        data_main['category'] = [cate_map[i] for i in main_idx]
    if l2_cate_map is not None:
        data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
    if split_map is not None:
        data_main['split'] = [split_map[i] for i in indices]

    # load split
    dump(data_main, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
    data_main = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

    acc = report_acc(data_main)
    score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
    dump(acc, score_file)
    logger.info(f'multiple_choice_eval successfully finished evaluating {eval_file}, results saved in {score_file}')
    logger.info('Score: ')
    logger.info(acc)
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM Answers. ')
    parser.add_argument('data', type=str, help='The question set for inference, in excel / tsv / json format. ')
    parser.add_argument(
        '--model',
        type=str,
        help='The LLM (GPT) used for inference. ',
        default='chatgpt-0613',
        choices=['chatgpt-0613', 'exact_matching', 'gpt-4-0125'])
    parser.add_argument(
        '--dataset',
        type=str,
        default='default',
        help='The dataset to evaluate')
    parser.add_argument('--nproc', type=int, default=6)
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
    acc = multiple_choice_eval(eval_file=args.data, dataset=args.dataset, **judge_kwargs)
