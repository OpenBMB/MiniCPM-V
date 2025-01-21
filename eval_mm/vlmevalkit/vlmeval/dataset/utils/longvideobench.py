from ...smp import *
from .multiple_choice import extract_answer_from_item
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

DURATIONS = [15, 60, 600, 3600]
TASK_CATEGORIES = [
    "S2E", "S2O", "S2A",
    "E2O", "O2E", "T2E",
    "T2O", "T2A", "E3E",
    "O3O", "SSS", "SOS",
    "SAA", "T3E", "T3O",
    "TOS", "TAA"
]


def get_dimension_rating(data_path):
    data = load(data_path)
    print(data.iloc[0])

    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ['overall']:
        duration_rating[duration] = {
            'overall': '',
            'question_category': {k: [] for k in TASK_CATEGORIES}
        }

    for i in range(len(data)):

        task_ctg = data.iloc[i]['question_category']

        duration = data.iloc[i]['duration_group']
        duration_rating[duration]['question_category'][task_ctg].append(data.iloc[i]['score'])

        duration_rating['overall']['question_category'][task_ctg].append(data.iloc[i]['score'])

    for duration in DURATIONS + ['overall']:
        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["question_category"].values(), []) if x >= 0]):.3f}'  # noqa: E501
        duration_rating[duration]['overall'] = overall_res_dur
        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["question_category"][task_ctg] if x >= 0]):.3f}'  # noqa: E501
            duration_rating[duration]['question_category'][task_ctg] = task_res_dur

    return duration_rating


def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ''
    return matches[0]
