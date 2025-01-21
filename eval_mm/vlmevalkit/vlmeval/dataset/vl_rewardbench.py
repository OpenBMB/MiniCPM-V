from ast import literal_eval

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


LLM_PARSE_ANSWER_PROMPT = '''
You are given a pairwise judgement for two responses. Please return the better response according to the judgement.
Return the Answer X ONLY. e.g., Answer 1 or Answer 2.

Judgement: {judgement}
'''


PROMPT_TEMPLATE = '''\
You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions.
Please analyze the following image and question, then determine which of the two provided answers is better.

Question: {query}

Answer 1: {answer_0}

Answer 2: {answer_1}

Please evaluate both answers based on the following criteria:
1. Accuracy: How well does the answer align with the visual information in the image?
2. Completeness: Does the answer fully address all aspects of the question?
3. Clarity: Is the answer easy to understand and well-articulated?
4. Relevance: Does the answer directly relate to the question and the image?

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2).\
For example: Overall Judgment: Answer X is better.

Your response should be structured and detailed, \
demonstrating your understanding of both the visual and textual elements of the task.'''


def get_score(line, parsed_response, random_number):
    gt_ans = line['human_ranking'].index(0 if random_number == 0 else 1) + 1
    if 'Answer 1'.lower() in parsed_response.lower():
        pred = 1
    elif 'Answer 2'.lower() in parsed_response.lower():
        pred = 2
    else:  # failed
        pred = 'None'  # random.choice([1, 2])

    if pred == gt_ans:
        return 1.0
    else:
        return 0.0


def VLRewardBench_eval_answer(model, line):
    response = toliststr(line['response'])
    random_number = sum(len(res) for res in response) % 2

    prompt = LLM_PARSE_ANSWER_PROMPT.format(judgement=line['prediction'])
    messages = [dict(type='text', value=prompt)]

    resp = model.generate(messages)
    score = get_score(line, resp, random_number)

    if score is None:
        return 'Unknown'
    return score


class VLRewardBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'VL-RewardBench': 'https://huggingface.co/datasets/MMInstruction/VL-RewardBench/resolve/main/vl_rewardbench.tsv'
    }
    DATASET_MD5 = {'VL-RewardBench': '1d2676f4ab4a5f755019ec0af2b28189'}

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)  # save image to local
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        response = toliststr(line['response'])
        random_number = sum(len(res) for res in response) % 2
        if random_number == 1:
            # randomly shuffle the order of the responses
            response = response[::-1]
        query_prompt = PROMPT_TEMPLATE.format(
            query=question, answer_0=response[0], answer_1=response[1]
        )
        msgs = msgs + [dict(type='text', value=query_prompt)]
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            raw_data = VLRewardBench('VL-RewardBench').data
            data = load(eval_file)
            data['prediction'] = [str(x) for x in data['prediction']]
            data['human_ranking'] = [literal_eval(x) for x in raw_data['answer']]

            judge_kwargs['temperature'] = 0
            judge_kwargs['timeout'] = 60
            model = build_judge(max_tokens=128, **judge_kwargs)

            assert model.working(), (
                'VLRewardBench evaluation requires a working OPENAI API\n'
                + DEBUG_MESSAGE
            )

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    VLRewardBench_eval_answer,
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
            # data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        category_scores = defaultdict(lambda: 0)
        category_cnt = defaultdict(lambda: 0)
        scores = defaultdict(lambda: 0)
        for i in range(lt):
            item = data.iloc[i]
            category_scores[item['category']] += item['score']
            category_cnt[item['category']] += 1
        # calculate the average score for each category
        for k, v in category_scores.items():
            scores[k] = v / category_cnt[k]
        # calculate category macro accuracy (average across categories)
        scores['Macro Accuracy'] = sum(scores.values()) / len(scores)
        # calculate the total average score
        scores['Overall Consistency'] = sum(category_scores.values()) / lt

        scores = {k: [v] for k, v in scores.items()}
        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores
