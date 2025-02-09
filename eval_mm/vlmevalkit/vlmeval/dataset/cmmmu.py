from .image_base import ImageBaseDataset
import random
from collections import Counter
import os
import re
import tempfile
from ..smp import *


def get_multi_choice_prediction(response, all_choices, index2ans):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    candidates = []

    for choice in all_choices:  # (A) (B) (C) (D)
        # Add the choice to candidates each time it appears in the response
        candidates.extend([choice for _ in range(response.count(f'({choice})'))])

    if len(candidates) == 0:
        for choice in all_choices:  # A B C D
            # Similarly, add the choice for each occurrence
            candidates.extend([choice for _ in range(response.count(f'{choice}'))])

    if len(candidates) == 0 and len(response.split()) >= 1:
        for index, ans in index2ans.items():
            # Add index for each occurrence of ans in response
            candidates.extend([index for _ in range(response.count(ans))])

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) >= 1:
        for index, ans in index2ans.items():
            if ans in response:
                candidates.append(index)
                # index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        return random.choice(all_choices)
        # return ''
    else:
        # Count the occurrence of each candidate
        candidate_counts = Counter(candidates)

        # Select the most frequent candidates
        max_count = max(candidate_counts.values())
        most_frequent_candidates = [c for c in all_choices if candidate_counts.get(c, 0) == max_count]

        # Combine the most frequent candidates in ABCD order
        return ''.join(most_frequent_candidates)


def extract_numbers(string):
    # Pattern for numbers with Chinese commas
    pattern_commas = r'-?\d{1,3}(?:，\d{3})+'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without Chinese commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+)(?![eE][+-]?\d+)(?!，\d)'

    # Extract numbers with Chinese commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without Chinese commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def count_letters(string):
    return sum(c.isalpha() and 'a' <= c <= 'z' or 'A' <= c <= 'Z' for c in string)


def normalize_str(string, answer):
    # check if characters in the string

    # if number, numerize it.
    if string is None:
        return [string]
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        if len(string) > len(answer) + 20 or count_letters(string) > count_letters(answer) + 2:
            return []
        return [string]


def get_fill_blank_prediction(response, answer):
    """get the prediction from the generated response,
    return a list of predicted strings or numbers"""

    def get_key_subresponses(response):
        response = response.strip("。").strip()
        sub_responses = re.split(r'。|\n', response)
        indicators_of_keys = ['是', '为', '所以', '等于', '方案', '选择',
                              '正确答案', '因此', '最后', '答案', '结果']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i], answer))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_TF_prediction(response):
    """get the prediction from the generated response,
    return a list of predicted strings or numbers"""

    def get_key_subresponses(response):
        response = response.strip("。").strip()
        sub_responses = re.split(r'。|\n', response)
        indicators_of_keys = ['是', '为', '所以', '判断',
                              '陈述', '说法', '表达', '答案', '结果']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            shortest_key_response = None
            # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


class CMMMU(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'CMMMU_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/CMMMU_VAL.tsv'
    }

    DATASET_MD5 = {
        'CMMMU_VAL': 'b4727e2fce2415bf646379e60c11a726'
    }

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')

        if not osp.exists(result_file):
            data = load(eval_file)
            assert 'answer' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]

            correct_count = 0
            correct_category = {
                '技术与工程': [0, 0],
                '科学': [0, 0],
                '健康与医学': [0, 0],
                '商业': [0, 0],
                '艺术与设计': [0, 0],
                '人文社会科学': [0, 0],
            }

            for i in tqdm(data.iterrows()):
                line = i[1]
                correct_category[line['category']][0] += 1

                # Options
                if line['type'] == '选择':
                    index2ans = {
                        'A': line['option1'],
                        'B': line['option2'],
                        'C': line['option3'],
                        'D': line['option4']
                    }
                    fact_option = get_multi_choice_prediction(line['prediction'], ['A', 'B', 'C', 'D'], index2ans)
                    if fact_option == line['answer']:
                        correct_count += 1
                        correct_category[line['category']][1] += 1

                # Binary
                elif line['type'] == '判断':
                    positive_keywords = ['正确', '对', '准确', '肯定', '对的']
                    negative_keywords = ['不对', '错误', '不正确', '不准确', '不合适', '否定', '错的', '错']
                    ambiguous_keywords = ['对错', '是否正确', '否正确', '或者', '是否', '正确性', '对不']

                    def judge_similarity(pred_list, positive_keywords, negative_keywords):
                        positive_count = 0
                        negative_count = 0

                        for pred in pred_list:
                            if any(pos_word in pred for pos_word in positive_keywords):
                                positive_count += 1
                            elif any(neg_word in pred for neg_word in negative_keywords):
                                negative_count += 1

                        if positive_count > negative_count:
                            return "对"
                        elif negative_count > positive_count:
                            return "错"
                        else:
                            return random.choice(['对', '错'])

                    answer = get_TF_prediction(line['prediction'])
                    answer = [word for word in answer if not any(ambiguous in word for ambiguous in ambiguous_keywords)]
                    fact_answer = judge_similarity(answer, positive_keywords, negative_keywords)
                    if fact_answer == line['answer']:
                        correct_count += 1
                        correct_category[line['category']][1] += 1

                # Fill the Blank
                else:
                    norm_answers = normalize_str(line['answer'], line['answer'])
                    predicted_answer = get_fill_blank_prediction(line['prediction'], line['answer'])

                    for pred in predicted_answer:
                        # already normalized
                        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
                            for norm_ans in norm_answers:
                                # only see if the string answer in the string pred
                                # print(norm_ans, pred)
                                if isinstance(norm_ans, str) and norm_ans in pred:
                                    correct_count += 1
                                    correct_category[line['category']][1] += 1
                        else:  # it's a number
                            if pred in norm_answers:
                                correct_count += 1
                                correct_category[line['category']][1] += 1

            accuracyz = {}
            accuracyz['总准确率'] = correct_count / len(data)
            for i in correct_category.keys():
                accuracyz[i] = correct_category[i][1] / correct_category[i][0]

            accuracyz = d2df(accuracyz)
            accuracyz.round(10)
            dump(accuracyz, result_file)

        result = pd.read_csv(result_file)
        return result

    def build_prompt(self, line):
        if line['type'] == '选择':
            tgt_path = self.dump_image(line)
            question = line['question']
            options_prompt = 'Options:\n'

            for i in [['A', '1'], ['B', '2'], ['C', '3'], ['D', '4']]:
                options_prompt += i[0] + '. ' + line['option' + i[1]] + '\n'

            prompt = (f'问题: {question}\n' + options_prompt
                      + '请回答上述多项选择题，并选出正确选项。这些题目可能包括单选和多选题型。如果所提供的信息不足以确定一个明确的答案，那么请根据可用的数据和你的判断来选择最可能正确的选项。')

            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
            msgs.append(dict(type='text', value=prompt))

            return msgs

        elif line['type'] == '判断':
            msgs = super().build_prompt(line)
            assert msgs[-1]['type'] == 'text'
            msgs[-1]['value'] += '\n请回答上述判断题，并根据题目描述和所给的信息来判断问题中陈述的对错。如果信息不完整或不足以作出绝对判断，请运用你的逻辑推理和现有信息来做出最可能的判断。'
            return msgs

        else:
            msgs = super().build_prompt(line)
            assert msgs[-1]['type'] == 'text'
            msgs[-1]['value'] += '\n请回答上述填空题，并根据题目的要求和所提供的信息来给出最恰当的答案。如果信息不足以确切回答，那么请依据现有的数据和你的推理能力来填写最合理的答案。'
            return msgs
