import re
from functools import partial

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


SYSTEM_PROMPT = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user \
prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate \
which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any \
answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. \
You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly \
responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one \
interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than \
providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate \
to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing \
important information in the assistants' answers that would be beneficial to include when responding to the user \
prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""


PROMPT_TEMPLATE = """\
"<|User Prompt|>\n{question}

<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>
"""


REGEX_PATTERN = re.compile("\[\[([AB<>=]+)\]\]")  # noqa: W605


def get_score(judgement, pattern=REGEX_PATTERN):
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        return matches[0].strip("\n"), False
    else:
        return None, True


def WildVision_auxeval(model, line):
    config = dict(question=line['question'], answer_1=line['A'], answer_2=line['B'])
    prompt = PROMPT_TEMPLATE.format(**config)

    prefix = 'data:image/jpeg;base64,'
    img = prefix + line['image']

    messages = [
        dict(type='text', value=prompt),
        dict(type='image', value=img)
    ]

    retry = 2
    while retry:
        resp = model.generate(messages)
        score, try_again = get_score(resp)
        if not try_again:
            break
        retry -= 1

    if score is None:
        return 'Unknown'
    return score


class WildVision(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'WildVision': 'https://opencompass.openxlab.space/utils/VLMEval/WildVision.tsv'
    }
    DATASET_MD5 = {'WildVision': 'b38f80156d49411c594772866b0d0b52'}

    score_map = {
        'A>>B': -2,
        'A>B': -1,
        'A=B': 0,
        'B>A': 1,
        'B>>A': 2
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        # WildVision adopts text first
        msgs = [dict(type='text', value=question)] + msgs
        return msgs

    @classmethod
    def gen_eval_base(self, eval_file, b64_map):
        data = load(eval_file)
        data['B'] = data.pop('prediction')
        data['A'] = data.pop('claude3_sonnet')
        data['image'] = [b64_map[x] for x in data['index']]
        return data
        # rev = cp.deepcopy(data)
        # rev['A'] = data['B']
        # rev['B'] = data['A']
        # rev['index'] = [x + '_rev' for x in data['index']]
        # return pd.concat([data, rev], ignore_index=True)

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # We adopt pairwise evaluation (twice for a pair) for this dataset
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            raw_data = WildVision('WildVision').data
            b64_map = {x: y for x, y in zip(raw_data['index'], raw_data['image'])}
            data = self.gen_eval_base(eval_file, b64_map)

            judge_kwargs['system_prompt'] = SYSTEM_PROMPT
            judge_kwargs['temperature'] = 0
            judge_kwargs['img_detail'] = 'high'
            judge_kwargs['timeout'] = 300
            model = build_judge(max_tokens=4096, **judge_kwargs)

            assert model.working(), ('WildVision evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    WildVision_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = v

            data['score'] = [ans[idx] for idx in data['index']]
            data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        scores = defaultdict(lambda: 0)
        for i in range(lt):
            item = data.iloc[i]
            if item['score'] not in self.score_map:
                score = 0
            else:
                score = self.score_map[item['score']]
                if '_rev' in item['index']:
                    score = -score
            scores[score] += 1
        name_map = {
            2: 'Much Better',
            1: 'Better',
            0: 'Tie',
            -1: 'Worse',
            -2: 'Much Worse'
        }
        scores = {name_map[k]: v for k, v in scores.items()}
        much_better = scores.get('Much Better', 0)
        better = scores.get('Better', 0)
        worse = scores.get('Worse', 0)
        much_worse = scores.get('Much Worse', 0)
        scores['Reward'] = (
            100 * much_better + 50 * better - 50 * worse - 100 * much_worse
        ) / lt
        scores['Win Rate'] = (better + much_better) / lt
        scores = {k: [v] for k, v in scores.items()}
        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores
