from ..smp import *
from ..dataset.utils.judge_util import build_judge
from ..dataset.utils.multiple_choice import extract_answer_from_item
from .matching_util import can_infer
from .mp_util import track_progress_rich


def MMMU_result_transfer(result_path):
    res = {}
    result_data = load(result_path)
    mcq = result_data['A'].notna()
    lt = len(result_data)
    for i in range(lt):
        line = result_data.iloc[i]
        if mcq[i]:
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            prediction = line['prediction']
            infer_prediction = can_infer(prediction, options)
            res[line['id']] = infer_prediction
        else:
            res[line['id']] = line['prediction']
    result_json = result_path.replace('.xlsx', '.json')
    dump(res, result_json)
    return result_json


def MMTBench_result_transfer(eval_file, dataset='default', **judge_kwargs):
    logger = get_logger('Evaluation')
    nproc = judge_kwargs.pop('nproc', 4)

    rd.seed(2680)
    suffix = eval_file.split('.')[-1]
    model = judge_kwargs['model']
    assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
    name_str_map = {
        'chatgpt-0125': 'openai',
        'gpt-4-0125': 'gpt4'
    }
    name_str = name_str_map[model] if model in name_str_map else model

    if model == 'exact_matching':
        model = None
    elif gpt_key_set():
        model = build_judge(**judge_kwargs)
        if not model.working():
            logger.error('The OPENAI API is not working properly, will use exact matching for evaluation')
            model = None
    else:
        logger.error('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
        model = None

    logger.info(f'Evaluating {eval_file}')
    result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_option.pkl')
    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    data = load(eval_file)
    assert 'index' in data, 'Essentail columns missing in the eval_file.'

    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    idx2lines = {data.iloc[i]['index']: data.iloc[i] for i in range(len(data))}
    idx2lines = {k: v for k, v in idx2lines.items() if k not in result}

    indices = list(idx2lines.keys())
    lines = [idx2lines[i] for i in indices]
    tups = [(model, line) for line in lines]
    res = track_progress_rich(
        extract_answer_from_item,
        tups,
        nproc=nproc,
        chunksize=nproc,
        save=result_file,
        keys=indices)

    for i, r in zip(indices, res):
        if i in result:
            assert result[i]['opt'] == r['opt'] and result[i]['log'] == r['log']
        else:
            result[i] = r

    indices = list(data['index'])
    data['opt'] = [result[i]['opt'] for i in data['index']]
    data['log'] = [result[i]['log'] for i in data['index']]

    # load split
    output_path = eval_file.replace(f'.{suffix}', f'_{name_str}_submission.tsv')
    dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_submission.tsv'))
    return output_path
