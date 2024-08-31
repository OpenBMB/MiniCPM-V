from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


class TextMCQDataset(TextBaseDataset):
    TYPE = 'MCQ'

    DATASET_URL = {}

    DATASET_MD5 = {}

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []

        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc


class CustomTextMCQDataset(TextMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
