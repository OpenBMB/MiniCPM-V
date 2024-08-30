from ...smp import *
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'

system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""

MMV_DIMENSIONS = {
    'CP': ['Video Topic', 'Video Emotion', 'Video Scene', 'Video Style'],
    'FP-S': ['OCR', 'Object Recognition', 'Attribute Recognition', 'Event Recognition', 'Human Motion', 'Counting'],
    'FP-C': ['Spatial Relationship', 'Human-object Interaction', 'Human Interaction'],
    'HL': ['Hallucination'],
    'LR': ['Structuralized Image-Text Understanding', 'Mathematical Calculation'],
    'AR': ['Physical Property', 'Function Reasoning', 'Identity Reasoning'],
    'RR': ['Natural Relation', 'Physical Relation', 'Social Relation'],
    'CSR': ['Common Sense Reasoning'],
    'TR': ['Counterfactual Reasoning', 'Causal Reasoning', 'Future Prediction'],
}
L3_DIMS = []
for k, v in MMV_DIMENSIONS.items():
    L3_DIMS.extend(v)

MMV_DIMENSIONS['Perception'] = []
MMV_DIMENSIONS['Reasoning'] = []
MMV_DIMENSIONS['Overall'] = []
for k in ['CP', 'FP-C', 'FP-S', 'HL']:
    MMV_DIMENSIONS['Perception'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])
for k in ['LR', 'AR', 'RR', 'CSR', 'TR']:
    MMV_DIMENSIONS['Reasoning'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])


def get_dimension_rating(data_path):
    data = load(data_path)
    coarse_rating = {k: [] for k in MMV_DIMENSIONS}
    fine_rating = {k: [] for k in L3_DIMS}

    for i in range(len(data)):
        cate = data.iloc[i]['dimensions']
        cates = eval(cate)

        for c in cates:
            fine_rating[c].append(data.iloc[i]['score'])

        for d in MMV_DIMENSIONS:
            if np.any([x in MMV_DIMENSIONS[d] for x in cates]):
                coarse_rating[d].append(data.iloc[i]['score'])

    coarse_all = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in coarse_rating.items()}
    coarse_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in coarse_rating.items()}
    fine_all = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in fine_rating.items()}
    fine_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in fine_rating.items()}
    return dict(coarse_all=coarse_all, coarse_valid=coarse_valid, fine_all=fine_all, fine_valid=fine_valid)


def build_prompt(item):
    tmpl = 'Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '
    return tmpl.format(item['question'], item['answer'], item['prediction'])
