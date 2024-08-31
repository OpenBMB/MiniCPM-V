from ...smp import *
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

DURATIONS = [
    'short',
    'medium',
    'long',
]

DOMAINS = [
    'Knowledge',
    'Film & Television',
    'Sports Competition',
    'Artistic Performance',
    'Life Record',
    'Multilingual'
]

SUB_CATEGORIES = [
    'Humanity & History',
    'Literature & Art',
    'Biology & Medicine',
    'Finance & Commerce',
    'Astronomy',
    'Geography',
    'Law',
    'Life Tip',
    'Technology',
    'Animation',
    'Movie & TV Show',
    'Documentary',
    'News Report',
    'Esports',
    'Basketball',
    'Football',
    'Athletics',
    'Other Sports',
    'Stage Play',
    'Magic Show',
    'Variety Show',
    'Acrobatics',
    'Handicraft',
    'Food',
    'Fashion',
    'Daily Life',
    'Travel',
    'Pet & Animal',
    'Exercise',
    'Multilingual'
]

TASK_CATEGORIES = [
    'Temporal Perception',
    'Spatial Perception',
    'Attribute Perception',
    'Action Recognition',
    'Object Recognition',
    'OCR Problems',
    'Counting Problem',
    'Temporal Reasoning',
    'Spatial Reasoning',
    'Action Reasoning',
    'Object Reasoning',
    'Information Synopsis',
]


def get_dimension_rating(data_path):
    data = load(data_path)

    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ['overall']:
        duration_rating[duration] = {
            'overall': '',
            'domain': {k: [] for k in DOMAINS},
            'sub_category': {k: [] for k in SUB_CATEGORIES},
            'task_type': {k: [] for k in TASK_CATEGORIES}
        }

    for i in range(len(data)):

        domain = data.iloc[i]['domain']
        sub_ctg = data.iloc[i]['sub_category']
        task_ctg = data.iloc[i]['task_type']

        duration = data.iloc[i]['duration']
        duration_rating[duration]['domain'][domain].append(data.iloc[i]['score'])
        duration_rating[duration]['sub_category'][sub_ctg].append(data.iloc[i]['score'])
        duration_rating[duration]['task_type'][task_ctg].append(data.iloc[i]['score'])

        duration_rating['overall']['domain'][domain].append(data.iloc[i]['score'])
        duration_rating['overall']['sub_category'][sub_ctg].append(data.iloc[i]['score'])
        duration_rating['overall']['task_type'][task_ctg].append(data.iloc[i]['score'])

    for duration in DURATIONS + ['overall']:

        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.2f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.2f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.2f}'
            duration_rating[duration]['sub_category'][sub_ctg] = sub_res_dur

        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.2f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

    return duration_rating


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

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]
