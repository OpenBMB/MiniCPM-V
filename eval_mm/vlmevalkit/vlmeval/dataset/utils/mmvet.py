from ...smp import *


def build_mmvet_gpt4_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    prompt = """
Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.
<AND> in the ground truth means it is totally right
only when all elements in the ground truth are present in the prediction,
and <OR> means it is totally right when any one element in the ground truth is present in the prediction.
The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries
Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes,
while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues
because the names of these countries do not accurately represent their landscapes. |
The meme talks about Iceland and Greenland. It's pointing out that despite their names,
Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries
Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes,
while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues
because the names of these countries do not accurately represent their landscapes. |
The meme is using humor to point out the misleading nature of Iceland's and Greenland's names.
Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow.
The text 'This is why I have trust issues' is a playful way to suggest
that these contradictions can lead to distrust or confusion.
The humor in this meme is derived from the unexpected contrast between the names of the countries
and their actual physical characteristics. | 1.0
"""
    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt


def MMVet_auxeval(model, line):
    def float_cvt(s):
        try:
            return float(s)
        except ValueError:
            return None

    prompt = build_mmvet_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 1:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, score=0.0)


def MMVet_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate2_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        grade = float(item['score'])
        cate_list = ['rec', 'ocr', 'know', 'gen', 'spat', 'math']
        for capa in cate_list:
            if capa in cate:
                tot[capa] += 1
                score[capa] += grade
        tot['Overall'] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate2] += grade

    res = defaultdict(list)
    res2 = defaultdict(list)
    cate_list.append('Overall')
    cate2_list.append('Overall')
    for k in cate_list:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 100)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    return res, res2
