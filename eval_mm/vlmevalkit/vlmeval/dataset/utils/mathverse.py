from ...smp import *
from ...utils import can_infer


FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4]


def build_mathverse_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt


def build_mathverse_gpt4_score_prompt(line):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    question_for_eval = line['question_for_eval']
    extract = line['extract']
    answer = line['answer']
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def post_check_score(line, prefetch=False):
    ans = str(line['answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False


def MathVerse_auxeval_extract(model, line):
    prompt = build_mathverse_gpt4_extract_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_extract=log, extract=res)
    log += 'All 5 retries failed.\n'
    return dict(log_extract=log, extract='')


def MathVerse_auxeval_score(model, line):
    prompt = build_mathverse_gpt4_score_prompt(line)
    log = ''
    retry = 5
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score='Prefetch succeed', score=True)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res or res.strip() not in ['0', '1']:
            log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_score=log, score=int(res) == 1)
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=False)


def MathVerse_acc(result_file):
    df = load(result_file)

    df['metadata'] = df['metadata'].apply(lambda x: x.replace("'", '"'))
    df['metadata'] = df['metadata'].apply(json.loads)
    df_metadata = pd.json_normalize(df['metadata'])
    df = pd.concat([df.drop('metadata', axis=1), df_metadata], axis=1)

    subset = list(set(df['problem_version']))

    res = defaultdict(list)
    for p in subset:
        if p != 'Overall':
            sub = df[df['problem_version'] == p]
        else:
            sub = cp.deepcopy(df)
        res['split'].append(p)
        # Overall Acc
        res['Overall'].append(np.mean(sub['score']) * 100)
        # Subject
        subjects = set(df['subject'])
        for k in subjects:
            res[k].append(np.mean(sub[sub['subject'] == k]['score']) * 100)
        # Subfield
        subfields = set(df['subfield'])
        for k in subfields:
            res[k].append(np.mean(sub[sub['subfield'] == k]['score']) * 100)

    return pd.DataFrame(res)
