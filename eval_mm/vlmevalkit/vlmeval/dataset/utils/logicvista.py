import pandas as pd

# from colorama import Fore, Back, Style
from ...smp import *


FAIL_MSG = 'Failed to obtain answer via API.'


def build_prompt_logicvista(line):
    question = line['question']
    prediction = str(line['prediction'])
    tmpl = (
        "You are a information extractor that extracts multiple choice letter answer choices "
        "from a paragraph that contains the answer choice and sometimes explaination of why that "
        "choice is correct to the given question.\n"
        "What letter did the following answer choose? If the answer did not select a letter answer choice, "
        "first try to infer the answer based off the given choices.\n"
        "If it does not seem like the given answer corresponds to an answer choice OR if there is no selected answer, please just respond with Z.\n"
        "Make sure you answer with ONLY the letters chosen.\n"
        'Example 1: \n'
        'Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n'
        'Answer: <start>\na cute teddy bear\n<end>\nYour output: A\n'
        'Example 2: \n'
        'Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n'
        'Answer: <start>\nSpider\n<end>\nYour output: Z\n'
        'Example 3: \n'
        'Question: <start>\nWhich figure is a rotation of the object?\n<end>\n'
        'Answer: <start>\nThe figure on the right, labeled "D," is a rotation of the object shown in the top left corner.\n<end>\nYour output: D\n'
        'Example 4: \n'
        'Question: <start>\nWhich of the boxes comes next in the sequence? Select from A-E\n<end>\n'
        'Answer: <start>\nThe sequence of the boxes is A, B, C, D, E.\n<end>\nYour output: ABCDE\n'
        'Example 5: \n'
        'Question: <start>\n{}\n<end>\nAnswer: <start>\n{}\n<end>\nYour output: '
    )

    return tmpl.format(question, prediction)


def LogicVista_auxeval(model, line):
    prompt = build_prompt_logicvista(line)
    print(prompt)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)
        answer = line['answer'].split(", ")
        for j in range(0, len(answer)):
            answer[j] = answer[j].lower()
        answer.sort()
        answer = ''.join(answer)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        elif not res.isupper() or not res.isalpha():
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            hit = 0
            extracted = [alpha.lower() for alpha in res]
            extracted.sort()
            extracted = ''.join(extracted)
            if extracted == answer:
                hit = 1
            return dict(log=log, res=res, hit=hit)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='', hit=0)


cat = ["diagram", "ocr", "patterns", "graphs", "tables", "3d shapes", "puzzles", "sequences", "physics"]


def evaluate_logicvista(file_path):
    df = pd.read_excel(file_path)

    tot = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    acc = defaultdict(lambda: 0)

    lt = len(df)
    skill_list = []

    df_tot = df

    df_inductive = df[df["skill"].str.contains("inductive")]
    df_deductive = df[df["skill"].str.contains("deductive")]
    df_numerical = df[df["skill"].str.contains("numerical")]
    df_spatial = df[df["skill"].str.contains("spatial")]
    df_mechanical = df[df["skill"].str.contains("mechanical")]

    tot_correct = df_tot["hit"].sum()
    tot_acc = (tot_correct / df_tot.shape[0]) * 100
    tot['Overall'] = df_tot.shape[0]
    hit['Overall'] = tot_correct
    acc['Overall'] = tot_acc

    inductive_correct = df_inductive["hit"].sum()
    inductive_acc = (inductive_correct / df_inductive.shape[0]) * 100

    tot["inductive"] = df_inductive.shape[0]
    hit["inductive"] = inductive_correct
    acc["inductive"] = inductive_acc

    deductive_correct = df_deductive["hit"].sum()
    deductive_acc = (deductive_correct / df_deductive.shape[0]) * 100

    tot["deductive"] = df_deductive.shape[0]
    hit["deductive"] = deductive_correct
    acc["deductive"] = deductive_acc

    numerical_correct = df_numerical["hit"].sum()
    numerical_acc = (numerical_correct / df_numerical.shape[0]) * 100

    tot["numerical"] = df_numerical.shape[0]
    hit["numerical"] = numerical_correct
    acc["numerical"] = numerical_acc

    spatial_correct = df_spatial["hit"].sum()
    spatial_acc = (spatial_correct / df_spatial.shape[0]) * 100

    tot["spatial"] = df_spatial.shape[0]
    hit["spatial"] = spatial_correct
    acc["spatial"] = spatial_acc

    mechanical_correct = df_mechanical["hit"].sum()
    mechanical_acc = (mechanical_correct / df_mechanical.shape[0]) * 100

    tot["mechanical"] = df_mechanical.shape[0]
    hit["mechanical"] = mechanical_correct
    acc["mechanical"] = mechanical_acc

    # capability dimension, the official data json does not contain 'capability' column, so it is now ignored
    # for i in cat:
    #     curr = df[df["capability"].str.contains(i.replace(" ", ""))]
    #     correct = curr["hit"].sum()
    #     accuracy = (correct / curr.shape[0]) * 100
    #     tot[i] = curr.shape[0]
    #     hit[i] = correct
    #     acc[i] = accuracy

    res = defaultdict(list)
    for k in tot.keys():
        res['Task&Skill'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(acc[k])
    res = pd.DataFrame(res)
    return res
