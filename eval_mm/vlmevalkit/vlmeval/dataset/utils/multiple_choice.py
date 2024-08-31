import pandas as pd
from ...utils import can_infer, track_progress_rich
from ...smp import *
import numpy as np

MMB_abbrs = {
    'coarse_perception': 'CP',
    'finegrained_perception (instance-level)': 'FP-S',
    'finegrained_perception (cross-instance)': 'FP-C',
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}

MMT_abbrs = {
    'visual_recognition': 'VR',
    'localization': 'Loc',
    'ocr': 'OCR',
    'counting': 'Count',
    'hallucination': 'HLN',
    'image_retrieval': 'IR',
    'threed': '3D',
    'visual_captioning': 'VC',
    'visual_grounding': 'VG',
    'doc_understanding': 'DU',
    'action_recognition': 'AR',
    'pixel_level_perception': 'PLP',
    'image-to-image_translation': 'I2IT',
    'relation_reasoning': 'RR',
    'intelligence_quotient_test': 'IQT',
    'emotion': 'Emo',
    'visual_illusion': 'VI',
    'meme_understanding': 'MemU',
    'visual_prompt_understanding': 'VPU',
    'anomaly_detection': 'AND',
    'keypoint_detection': 'KD',
    'visual_commonsense_reasoning': 'VCR',
    'image_evaluation_judgement': 'IEJ',
    'multiple_image_analysis': 'MIA',
    'cross_image_matching': 'CIM',
    'temporal_understanding': 'TU',
    'visual_code': 'VP',
    'medical_understanding': 'MedU',
    'autonomous_driving': 'AUD',
    'discipline_knowledge_reasoning': 'DKR',
    'embodied_ai': 'EA',
    'gui_navigation': 'GN'
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
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
    return pd.DataFrame(res)


def report_acc_MMT(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)
    res['split'] = list()
    res['Overall'] = list()
    for _, name in MMT_abbrs.items():
        res[name] = list()

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits

    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'category', 'l2-category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
            res['Overall'].extend([np.mean(df['hit'])])
        elif group not in df:
            continue
        elif group == 'category':
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
                res[ab_name].extend([np.mean(sub_df['hit'])])
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                sub_task_name_list = df[df['l2-category'] == ab]['category'].unique()
                sub_task_acc = []
                for sub_task_name in sub_task_name_list:
                    sub_df = df[df['category'] == sub_task_name]
                    sub_task_acc.append([np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']])

                new_acc = []
                for i in range(len(sub_task_acc[0])):
                    new_acc.append(sum([_[i] for _ in sub_task_acc]) / len([_ for _ in sub_task_acc]))
                ab_name = MMT_abbrs[ab] if ab in MMT_abbrs else ab
                res[ab_name] = new_acc

                sub_task_acc = []
                for sub_task_name in sub_task_name_list:
                    sub_df = df[df['category'] == sub_task_name]
                    sub_task_acc.append([np.mean(sub_df['hit'])])
                new_acc = []
                for i in range(len(sub_task_acc[0])):
                    new_acc.append(sum([_[i] for _ in sub_task_acc]) / len([_ for _ in sub_task_acc]))

                res[ab_name].extend(new_acc)

    res['split'].append('ALL')
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


def build_prompt_blink(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output Z."
        'If the meaning of all options are significantly different from the answer, '
        'or the answer does not select any option, output Z. '
        'Your should output one of the choices, A, B, C, D (if they are valid options), or Z.\n'
        'Example 1: \n'
        'Question: Which point is closer to the camera?\nSelect from the following choices.\n'
        'Options: A. Point A\nB. Point B\n(Z) Failed\n'
        'Answer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n'
        'Example 2: \n'
        'Question: Which point is closer to the camera?\nSelect from the following choices.\n'
        'Options: (A) Point A\n(B) Point B\n(Z) Failed\n'
        "Answer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        'Example 3: \n'
        'Question: Which point is corresponding to the reference point?\nSelect from the following choices.\n'
        'Options: (A) Point A\n(B) Point B\n(Z) Failed\n'
        'Answer:The reference point (REF) on the first image is at the tip of the pot, '
        'which is the part used to Poke if the pots were used for that action. Looking at the second image, '
        'we need to find the part of the object that would correspond to poking.\n'
        "(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n"
        '(B) Point B is at the bottom of the spoon, which is not used for poking.\n'
        '(C) Point C is on the side of the pspoonot, which is not used for poking.\n'
        '(D) Point D is at the tip of the spoon, which is not used for poking.\n'
        '\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n'
        'Example 4: \n'
        'Question: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: '
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


def extract_answer_from_item(model, item, dataset_name=None):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if dataset_name == 'BLINK':
        prompt = build_prompt_blink(item['question'], option_str, item['prediction'])
    elif cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 3

    ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])
    if model is None:
        return dict(opt='Z', log='Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.')

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


# For Circular Evaluation
def prefetch_circular_group(sub_data, verbose=False):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        GT.append(item['GT'])
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


def eval_vanilla(model, item, dataset_name=None):
    res = extract_answer_from_item(model, item, dataset_name=dataset_name)
    opt, match_log = res['opt'], res['log']
    if opt == item['GT']:
        return dict(hit=1, log=f'Match Log: {match_log}. ')
    else:
        return dict(hit=0, log=f'Match Log: {match_log}. ')


# For Circular Evaluation
def eval_circular_group(model, sub_data, dataset_name=None):
    res, GT, PRED = prefetch_circular_group(sub_data, verbose=True)
    if res is not None:
        return res

    lt = len(sub_data)
    log = ''
    for i in range(lt):
        if PRED[i]:
            log += f'Rolling {i} Matched.\n'
        else:
            res = extract_answer_from_item(model, sub_data.iloc[i], dataset_name=dataset_name)
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


# data, meta are pd.DataFrame, result_file is a path
def mcq_vanilla_eval(model, data, meta, nproc, result_file, dataset_name=None):
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

    if 'MMMU' in dataset_name:
        data = MMMU_preproc(data)
        answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}

    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in items]
    keys = [x['index'] for x in items]
    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=nproc, chunksize=nproc, save=result_file, keys=keys)
        result = load(result_file)
        for k, v in zip(keys, res):
            if k in result:
                assert result[k]['hit'] == v['hit'] and result[k]['log'] == v['log']
            else:
                result[k] = v
    data['hit'] = [result[i]['hit'] for i in data['index']]
    data['log'] = [result[i]['log'] for i in data['index']]
    if 'GT' in data:
        data.pop('GT')
    return data


# data, meta are pd.DataFrame, result_file is a path
def mcq_circular_eval(model, data, meta, nproc, result_file, dataset_name=None):
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
    # Build Answer Map
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

    for idx in list(meta['index']) + list(data['index']):
        assert istype(idx, int)

    # Only keep those lines in the meta data
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    data_main = data[data['index'] < int(1e6)]

    data_groups = []
    for i in range(len(data_main)):
        # Dealing with the normal part
        idx = data_main.iloc[i]['index']
        if idx not in result:
            sub_data = data[data['index'] % int(1e6) == idx]
            data_groups.append(sub_data)

    if len(data_groups):
        prefetched = [prefetch_circular_group(g, verbose=False) for g in data_groups]
        remain = []
        for dg, pf in zip(data_groups, prefetched):
            if pf is not None:
                result[dg.iloc[0]['index'] % 1e6] = pf
            else:
                remain.append(dg)
        dump(result, result_file)

        tups = [dict(model=model, sub_data=x, dataset_name=dataset_name) for x in remain]
        keys = [x.iloc[0]['index'] % 1e6 for x in remain]

        if len(tups) == 0:
            pass
        elif model is None:
            logger = get_logger('Evaluation')
            logger.warning('Exact Matching mode, will not do GPT-based answer matching. ')
            for k in keys:
                result[k] = dict(
                    hit=0, log='Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.')
        else:
            res = track_progress_rich(
                eval_circular_group,
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

    tmp_pth = f'/tmp/{timestr()}.xlsx'
    dump(data_main, tmp_pth)
    data_main = load(tmp_pth)
    indices = data_main['index']
    data_main['hit'] = [result[i]['hit'] for i in indices]
    data_main['log'] = [result[i]['log'] for i in indices]
    if 'GT' in data_main:
        data_main.pop('GT')

    return data_main
