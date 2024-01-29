import argparse
import json
import random
import os

import tqdm
import time
import json
import re
import string
import sys
from collections import Counter
from pycocoevalcap.rouge.rouge import Rouge
import editdistance
import pandas as pd


def ANLS(pred, answers):
    scores = []
    for ans in answers:
        a, b = ans.lower().split(), pred.lower().split()
        ed = editdistance.eval(a, b)
        NL = ed / max(len(a), len(b))
        scores.append(1 - NL)
        # scores.append(1 - NL if NL < 0.5 else 0)
    return max(scores)


def rouge_score(prediction, references):
    rouge = Rouge()
    rouge.beta = 10

    references = [' '.join(normalize_answer(x).split()) for x in references]
    prediction = ' '.join(normalize_answer(prediction).split())

    rouge_score = rouge.calc_score([prediction], references)
    return rouge_score


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction, groudtruths):
    scores = [_get_f1_score(prediction, ground_truth)
              for ground_truth in groudtruths]
    p = max([s[0] for s in scores])
    r = max([s[1] for s in scores])
    f = max([s[2] for s in scores])
    return p, r, f


def _get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question")
    parser.add_argument("-a", "--answer")
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    questions = json.load(f_q)
    category_lst = ["AOKVQA", "GQA", "OKVQA",
                    "ScienceQA", "VizWiz", "VQAv2", "WebQA"]
    for i, ques_js in enumerate(questions):
        ques_js["category"] = category_lst[i // 100]
    ans1_list = [json.loads(line)
                 for line in open(os.path.expanduser(args.answer))]

    idx = 0
    scores = []
    for ques, ans1 in zip(questions, ans1_list):
        category = ques["category"]
        if category == "AOKVQA":
            answers = (
                ques["Human Answers"].replace(
                    "[", "").replace("]", "").split("\n")
            )
        elif category == "GQA":
            answers = ques["Human Answers"].split("\n")
        elif category == "OKVQA":
            answers = (
                ques["Human Answers"]
                .replace("]", "")
                .replace("Human answers are: [", "")
            ).split("\n")

            def okvqa_clean_answer(string):
                i = string.index("with a confidence level of")
                string = string[:i]
                string = string.replace('"', "").strip()
                return string

            answers = [okvqa_clean_answer(ans) for ans in answers]
        elif category == "ScienceQA":
            answers = ques["Human Answers"].split("\n")
        elif category == "VizWiz":
            answers = (
                ques["Human Answers"]
                .replace("Human answers are:", "")
                .replace("[", "")
                .replace("]", "")
                .split("\n")
            )

            def vizwiz_clean_answer(string):
                i = string.index("with a confidence level of")
                string = string[:i]
                string = string.replace('"', "").strip()
                return string

            answers = [vizwiz_clean_answer(ans) for ans in answers]
        elif category == "VQAv2":
            answers = (
                ques["Human Answers"].replace(
                    "[", "").replace("]", "").split("\n")
            )

            def vqa_clean_answer(string):
                i = string.index("with a confidence level of")
                string = string[:i]
                string = (
                    string.replace("one of the human answers is", "")
                    .replace('"', "")
                    .strip()
                )
                return string

            answers = [vqa_clean_answer(ans) for ans in answers]
        elif category == "WebQA":
            answers = ques["answer"].split("\n")
        prediction = ans1["text"]
        _rouge_score = rouge_score(prediction, answers)
        # if random.random() > 0.8:
        #     print(prediction, answers, _rouge_score)
        _precision_score, _recall_score, _f1_score = get_f1_score(
            prediction, answers)
        _anls = ANLS(prediction, answers)
        if category in ["ScienceQA", 'WebQA', 'VizWiz']:
            continue
        scores.append(
            {
                "rouge": _rouge_score,
                "precision": _precision_score,
                "recall": _recall_score,
                "f1": _f1_score,
                "anls": _anls,
                "category": category,
            }
        )
    scores = pd.DataFrame(scores)
    category_scores = scores.groupby("category").mean()
    total_score = pd.DataFrame(
        scores[['rouge', 'precision', 'recall', 'f1', 'anls']].mean(), columns=['total']).T
    category_scores = pd.concat([category_scores, total_score])
    print(category_scores)
    # category_scores.to_csv(f'{args.answer}.csv')
