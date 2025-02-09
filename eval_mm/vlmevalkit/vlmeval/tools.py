import sys
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.config import *
from vlmeval.smp import *

# Define valid modes
MODES = ('dlist', 'mlist', 'missing', 'circular', 'localize', 'check', 'run', 'eval', 'merge_pkl')

CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['vlmutil'] + sys.argv[1:])}. vlmutil commands use the following syntax:

        vlmutil MODE MODE_ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode

    Some usages for xtuner commands: (See more by using -h for specific command!)

        1. List all the dataset by levels: l1, l2, l3, etc.:
            vlmutil dlist [l1/l2/l3/...]
        2. List all the models by categories: 4.33.0, 4.37.0, api, etc.:
            vlmutil mlist 4.33.0 [all/small/large]
        3. Report missing results:
            vlmutil missing [l1/l2/l3/...]
        4. Create circular questions (only for multiple-choice questions with no more than 4 choices):
            vlmutil circular input.tsv
        5. Create a localized version of the dataset (for very large tsv files):
            vlmutil localize input.tsv
        6. Check the validity of a model:
            vlmutil check [model_name/model_series]
        7. Run evaluation for missing results:
            vlmutil run l2 hf
        8. Evaluate data file:
            vlmutil eval [dataset_name] [prediction_file]
        9. Merge pkl files:
            vlmutil merge_pkl [pkl_dir] [world_size]

    GitHub: https://github.com/open-compass/VLMEvalKit
    """  # noqa: E501


dataset_levels = {
    'l1': [
        ('MMVet', 'gpt-4-turbo_score.csv'), ('MMMU_DEV_VAL', 'acc.csv'),
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('HallusionBench', 'score.csv'),
        ('OCRBench', 'score.json'), ('AI2D_TEST', 'acc.csv'), ('MMStar', 'acc.csv'),
        ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv')
    ],
    'l2': [
        ('MME', 'score.csv'), ('LLaVABench', 'score.csv'), ('RealWorldQA', 'acc.csv'),
        ('MMBench', 'acc.csv'), ('MMBench_CN', 'acc.csv'), ('CCBench', 'acc.csv'),
        ('SEEDBench_IMG', 'acc.csv'), ('COCO_VAL', 'score.json'), ('POPE', 'score.csv'),
        ('ScienceQA_VAL', 'acc.csv'), ('ScienceQA_TEST', 'acc.csv'), ('MMT-Bench_VAL', 'acc.csv'),
        ('SEEDBench2_Plus', 'acc.csv'), ('BLINK', 'acc.csv'), ('MTVQA_TEST', 'acc.json'),
        ('Q-Bench1_VAL', 'acc.csv'), ('A-Bench_VAL', 'acc.csv'), ('R-Bench-Dis', 'acc.csv'),
        ('MathVision', 'score.csv'), ('MathVerse_MINI_Vision_Only', 'score.csv'), ('DynaMath', 'score.csv'),
    ],
    'l3': [
        ('OCRVQA_TESTCORE', 'acc.csv'), ('TextVQA_VAL', 'acc.csv'),
        ('ChartQA_TEST', 'acc.csv'), ('DocVQA_VAL', 'acc.csv'), ('InfoVQA_VAL', 'acc.csv'),
        ('SEEDBench2', 'acc.csv')
    ]
}

dataset_levels['l12'] = dataset_levels['l1'] + dataset_levels['l2']
dataset_levels['l23'] = dataset_levels['l2'] + dataset_levels['l3']
dataset_levels['l123'] = dataset_levels['l12'] + dataset_levels['l3']

models = {
    '4.37.0': ['MiniCPM-V', 'MiniCPM-V-2'],
    '4.40.0': ['MiniCPM-Llama3-V-2_5'],
    'latest': ['MiniCPM-V-2_6']
}

# SKIP_MODELS will be skipped in report_missing and run APIs
SKIP_MODELS = ['MiniCPM-V']

def completed(m, d, suf):
    score_file = f'outputs/{m}/{m}_{d}_{suf}'
    if osp.exists(score_file):
        return True
    if d == 'MMBench':
        s1, s2 = f'outputs/{m}/{m}_MMBench_DEV_EN_{suf}', f'outputs/{m}/{m}_MMBench_TEST_EN_{suf}'
        return osp.exists(s1) and osp.exists(s2)
    elif d == 'MMBench_CN':
        s1, s2 = f'outputs/{m}/{m}_MMBench_DEV_CN_{suf}', f'outputs/{m}/{m}_MMBench_TEST_CN_{suf}'
        return osp.exists(s1) and osp.exists(s2)
    return False


def DLIST(lvl):
    if lvl in dataset_levels.keys():
        return [x[0] for x in dataset_levels[lvl]]
    else:
        from vlmeval.dataset import SUPPORTED_DATASETS
        return SUPPORTED_DATASETS


def MLIST(lvl, size='all'):
    if lvl == 'all':
        from vlmeval.config import supported_VLM
        return [x for x in supported_VLM]

    model_list = models[lvl]
    if size == 'small':
        model_list = [m for m in model_list if m not in LARGE_MODELS]
    elif size == 'large':
        model_list = [m for m in model_list if m in LARGE_MODELS]
    return [x[0] for x in model_list]


def MISSING(lvl):
    from vlmeval.config import supported_VLM
    models = list(supported_VLM)
    models = [m for m in models if m not in SKIP_MODELS and osp.exists(osp.join('outputs', m))]
    if lvl in dataset_levels.keys():
        data_list = dataset_levels[lvl]
    else:
        data_list = [(D, suff) for (D, suff) in dataset_levels['l123'] if D == lvl]
    missing_list = []
    for f in models:
        for D, suff in data_list:
            if not completed(f, D, suff):
                missing_list.append((f, D))
    return missing_list


def CIRCULAR(inp):
    assert inp.endswith('.tsv')
    data = load(inp)
    OFFSET = 1e6
    while max(data['index']) >= OFFSET:
        OFFSET *= 10

    assert 'E' not in data, 'Currently build_circular only works for up to 4-choice questions'
    data_2c = data[pd.isna(data['C'])]
    data_3c = data[~pd.isna(data['C']) & pd.isna(data['D'])]
    data_4c = data[~pd.isna(data['D'])]
    map_2c = [('AB', 'BA')]
    map_3c = [('ABC', 'BCA'), ('ABC', 'CAB')]
    map_4c = [('ABCD', 'BCDA'), ('ABCD', 'CDAB'), ('ABCD', 'DABC')]

    def okn(o, n=4):
        ostr = o.replace(',', ' ')
        osplits = ostr.split()
        if sum([c in osplits for c in string.ascii_uppercase[:n - 1]]) == n - 1:
            return False
        olower = o.lower()
        olower = olower.replace(',', ' ')
        olower_splits = olower.split()
        if 'all' in olower_splits or 'none' in olower_splits:
            return False
        return True

    yay4, nay4 = [], []
    lt4 = len(data_4c)
    for i in range(lt4):
        if okn(data_4c.iloc[i]['D'], 4):
            yay4.append(i)
        else:
            nay4.append(i)
    data_4c_y = data_4c.iloc[yay4]
    data_4c_n = data_4c.iloc[nay4]
    data_3c = pd.concat([data_4c_n, data_3c])

    yay3, nay3 = [], []
    lt3 = len(data_3c)
    for i in range(lt3):
        if okn(data_3c.iloc[i]['C'], 3):
            yay3.append(i)
        else:
            nay3.append(i)
    data_3c_y = data_3c.iloc[yay3]
    data_3c_n = data_3c.iloc[nay3]
    data_2c = pd.concat([data_3c_n, data_2c])

    def remap(data_in, tup, off):
        off = int(off)
        data = data_in.copy()
        char_map = {k: v for k, v in zip(*tup)}
        idx = data.pop('index')
        answer = data.pop('answer')
        answer_new = [char_map[x] if x in char_map else x for x in answer]
        data['answer'] = answer_new
        options = {}
        for c in char_map:
            options[char_map[c]] = data.pop(c)
        for c in options:
            data[c] = options[c]
        data.pop('image')
        data['image'] = idx
        idx = [x + off for x in idx]
        data['index'] = idx
        return data

    data_all = pd.concat([
        data_2c,
        data_3c_y,
        data_4c_y,
        remap(data_2c, map_2c[0], OFFSET),
        remap(data_3c_y, map_3c[0], OFFSET),
        remap(data_4c_y, map_4c[0], OFFSET),
        remap(data_3c_y, map_3c[1], OFFSET * 2),
        remap(data_4c_y, map_4c[1], OFFSET * 2),
        remap(data_4c_y, map_4c[2], OFFSET * 3),
    ])

    tgt_file = inp.replace('.tsv', '_CIRC.tsv')
    dump(data_all, tgt_file)
    print(f'The circularized data is saved to {tgt_file}')
    assert osp.exists(tgt_file)
    print(f'The MD5 for the circularized data is {md5(tgt_file)}')


PTH = osp.realpath(__file__)
IMAGE_PTH = osp.join(osp.dirname(PTH), '../assets/apple.jpg')

msg1 = [
    IMAGE_PTH,
    'What is in this image?'
]
msg2 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='What is in this image?')
]
msg3 = [
    IMAGE_PTH,
    IMAGE_PTH,
    'How many apples are there in these images?'
]
msg4 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='How many apples are there in these images?')
]


def CHECK(val):
    if val in supported_VLM:
        model = supported_VLM[val]()
        print(f'Model: {val}')
        for i, msg in enumerate([msg1, msg2, msg3, msg4]):
            if i > 1 and not model.INTERLEAVE:
                continue
            res = model.generate(msg)
            print(f'Test {i + 1}: {res}')
    elif val in models:
        model_list = models[val]
        for m in model_list:
            CHECK(m)


def LOCALIZE(fname, new_fname=None):
    if new_fname is None:
        new_fname = fname.replace('.tsv', '_local.tsv')

    base_name = osp.basename(fname)
    dname = osp.splitext(base_name)[0]

    data = load(fname)
    data_new = localize_df(data, dname)
    dump(data_new, new_fname)
    print(f'The localized version of data file is {new_fname}')
    return new_fname


def RUN(lvl, model):
    import torch
    NGPU = torch.cuda.device_count()
    SCRIPT = osp.join(osp.dirname(__file__), '../run.py')
    logger = get_logger('Run Missing')

    def get_env(name):
        assert name in ['433', '437', '440', 'latest']
        load_env()
        env_key = f'ENV_{name}'
        return os.environ.get(env_key, None)

    missing = MISSING(lvl)
    if model == 'all':
        pass
    elif model == 'api':
        missing = [x for x in missing if x[0] in models['api']]
    elif model == 'hf':
        missing = [x for x in missing if x[0] not in models['api']]
    elif model in models:
        missing = [x for x in missing if x[0] in models[missing]]
    elif model in supported_VLM:
        missing = [x for x in missing if x[0] == model]
    else:
        warnings.warn(f'Invalid model {model}.')

    missing.sort(key=lambda x: x[0])
    groups = defaultdict(list)
    for m, D in missing:
        groups[m].append(D)
    for m in groups:
        if m in SKIP_MODELS:
            continue
        for dataset in groups[m]:
            logger.info(f'Running {m} on {dataset}')
            exe = 'python' if m in LARGE_MODELS or m in models['api'] else 'torchrun'
            if m not in models['api']:
                env = None
                env = 'latest' if m in models['latest'] else env
                env = '433' if m in models['4.33.0'] else env
                env = '437' if m in models['4.37.0'] else env
                env = '440' if m in models['4.40.0'] else env
                if env is None:
                    # Not found, default to latest
                    env = 'latest'
                    logger.warning(
                        f"Model {m} does not have a specific environment configuration. Defaulting to 'latest'.")
                pth = get_env(env)
                if pth is not None:
                    exe = osp.join(pth, 'bin', exe)
                else:
                    logger.warning(f'Cannot find the env path {env} for model {m}')
            if exe.endswith('torchrun'):
                cmd = f'{exe} --nproc-per-node={NGPU} {SCRIPT} --model {m} --data {dataset}'
            elif exe.endswith('python'):
                cmd = f'{exe} {SCRIPT} --model {m} --data {dataset}'
            os.system(cmd)


def EVAL(dataset_name, data_file, **kwargs):
    from vlmeval.dataset import build_dataset
    logger = get_logger('VLMEvalKit Tool-Eval')
    dataset = build_dataset(dataset_name)
    # Set the judge kwargs first before evaluation or dumping
    judge_kwargs = {'nproc': 4, 'verbose': True}
    if 'model' not in kwargs:
        if dataset.TYPE in ['MCQ', 'Y/N']:
            judge_kwargs['model'] = 'chatgpt-0125'
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench-Video'], dataset_name):
            judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['MMLongBench', 'MMDU'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['DynaMath', 'MathVerse', 'MathVista', 'MathVision'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
    else:
        judge_kwargs['model'] = kwargs['model']
    judge_kwargs['nproc'] = kwargs.get('nproc', 4)
    eval_results = dataset.evaluate(data_file, **judge_kwargs)
    if eval_results is not None:
        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
        logger.info('Evaluation Results:')
    if isinstance(eval_results, dict):
        logger.info('\n' + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        logger.info('\n')
        logger.info(tabulate(eval_results.T) if len(eval_results) < len(eval_results.columns) else eval_results)
    return eval_results


def parse_args_eval():
    parser = argparse.ArgumentParser()
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('cmd', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('--judge', type=str, default=None)
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--retry', type=int, default=None)
    args = parser.parse_args()
    return args


def MERGE_PKL(pkl_dir, world_size=1):
    prefs = []
    for ws in list(range(1, 9)):
        prefs.extend([f'{i}{ws}_' for i in range(ws)])
    prefs = set(prefs)
    files = os.listdir(pkl_dir)
    files = [x for x in files if x[:3] in prefs]
    # Merge the files
    res_all = defaultdict(dict)
    for f in files:
        full_path = osp.join(pkl_dir, f)
        key = f[3:]
        res_all[key].update(load(full_path))
        os.remove(full_path)

    dump_prefs = [f'{i}{world_size}_' for i in range(world_size)]
    for k in res_all:
        for pf in dump_prefs:
            dump(res_all[k], f'{pkl_dir}/{pf}{k}')
        print(f'Merged {len(res_all[k])} records into {pkl_dir}/{dump_prefs[0]}{k}')


def cli():
    logger = get_logger('VLMEvalKit Tools')
    args = sys.argv[1:]
    if not args:  # no arguments passed
        logger.info(CLI_HELP_MSG)
        return

    if args[0].lower() == 'dlist':
        assert len(args) >= 2
        lst = DLIST(args[1])
        print(' '.join(lst))
    elif args[0].lower() == 'mlist':
        assert len(args) >= 2
        size = 'all'
        if len(args) > 2:
            size = args[2].lower()
        lst = MLIST(args[1], size)
        print('\n'.join(lst))
    elif args[0].lower() == 'missing':
        assert len(args) >= 2
        missing_list = MISSING(args[1])
        logger = get_logger('Find Missing')
        logger.info(colored(f'Level {args[1]} Missing Results: ', 'red'))
        lines = []
        for m, D in missing_list:
            line = f'Model {m}, Dataset {D}'
            logger.info(colored(line, 'red'))
            lines.append(line)
        mwlines(lines, f'{args[1]}_missing.txt')
    elif args[0].lower() == 'circular':
        assert len(args) >= 2
        CIRCULAR(args[1])
    elif args[0].lower() == 'localize':
        assert len(args) >= 2
        LOCALIZE(args[1])
    elif args[0].lower() == 'check':
        assert len(args) >= 2
        model_list = args[1:]
        for m in model_list:
            CHECK(m)
    elif args[0].lower() == 'run':
        assert len(args) >= 2
        lvl = args[1]
        if len(args) == 2:
            model = 'all'
            RUN(lvl, model)
        else:
            for model in args[2:]:
                RUN(lvl, model)
    elif args[0].lower() == 'eval':
        args = parse_args_eval()
        data_file = args.data_file

        def extract_dataset(file_name):
            fname = osp.splitext(file_name)[0].split('/')[-1]
            parts = fname.split('_')
            for i in range(len(parts)):
                if '_'.join(parts[i:]) in SUPPORTED_DATASETS:
                    return '_'.join(parts[i:])
            return None

        dataset = extract_dataset(data_file)
        assert dataset is not None, f'Cannot infer dataset name from {data_file}'
        kwargs = {'nproc': args.api_nproc}
        if args.judge is not None:
            kwargs['model'] = args.judge
        if args.retry is not None:
            kwargs['retry'] = args.retry
        EVAL(dataset_name=dataset, data_file=data_file, **kwargs)
    elif args[0].lower() == 'merge_pkl':
        assert len(args) == 3
        args[2] = int(args[2])
        assert args[2] in [1, 2, 4, 8]
        MERGE_PKL(args[1], args[2])
    else:
        logger.error('WARNING: command error!')
        logger.info(CLI_HELP_MSG)
        return
