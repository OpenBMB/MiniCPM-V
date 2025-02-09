from ...smp import *
from .multiple_choice import extract_answer_from_item
from PIL import Image, ImageOps
import numpy as np

sys_prompt = "You are an AI assistant for question answering."

system_prompt_multi_choice = (
    "You will receive a multi-choice question, the ground-truth answer and the prediction from a question answering (QA) model. "  # noqa
    "Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. "
    "If the prediction is correct, respond \"Correct\". If the prediction is incorrect, respond \"Incorrect\"."
)

system_prompt_caption_matching = (
    "You will receive a caption matching question, the ground-truth answer and the prediction from a question answering (QA) model. "  # noqa
    "Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. "
    "If the prediction is correct, respond \"Correct\". If the prediction is incorrect, respond \"Incorrect\"."
)

system_prompt_captioning = """
You will receive a video description and a multi-choice question. Your task is to choose the correct answer and briefly explain the reason why you choose the answer. \
If none of the choice candidates are correct or the video description lacks enough information to answer the question, just answer "None of the choices are correct". \
Please organize your response in this format:
```
Reasoning: [Your reason to obtain the answer]
Answer: [Your answer]
```

Here are some examples of video description, multi-choice question and the expected answer:
```
Video Description: A person is palying football.
Multi-Choice Question:
What is the person doing in the video?
A. cooking
B. palying football
C. playing basketball
D. reading book
Reasoning: The video description mentions that the person is playing football.
Answer: B. palying football

Video Description: A bird is flying clockwise.
Multi-Choice Question:
In which direction is the bird flying?
A. backwark
B. counter-clockwise
C. clockwise
D. downward
Reasoning: The video description mentions that the bird is flying clockwise
Answer: C. clockwise

Video Description: An air balloon is inflating.
Multi-Choice Question:
What is happening to the air balloon?
A. exploding
B. getting smaller
C. flying
Reasoning: The video description mentions that the air balloon is inflating, while none of the coices can be explained as inflating.
Answer: None of the choices are correct
```
"""  # noqa

system_prompt_YorN = """
You will receive a Yes/No question, the ground-truth answer and the prediction from a question answering (QA) model. \
Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
"""  # noqa


def eval_rule_caption_matching(line):
    # Determine whether the video llm output is correct, based on word matching rules
    video_llm_output = line['prediction']
    answer = line['answer']
    option_strs = eval(line['candidates'])  # complete option strings
    option_sents = [opt.split(': ')[1] for opt in option_strs]    # option sentence
    # option index, e.g., Sentence A, Caption A, Option 1
    option_inds = [opt.split(': ')[0] for opt in option_strs] + [opt.split(': ')[0].replace('Sentence ', '').replace('Option ', '').replace('Caption ', '') for opt in option_strs]  # noqa
    video_llm_pred = None
    for option_str in option_strs:
        if option_str == video_llm_output:
            video_llm_pred = option_str
    for option_sent in option_sents:
        if option_sent == video_llm_output or (') ' in video_llm_output and option_sent == video_llm_output.split(') ')[1]):  # noqa
            video_llm_pred = option_sent
    for option_ind in option_inds:
        if option_ind == video_llm_output or option_ind == video_llm_output.replace('.', ''):  # noqa
            video_llm_pred = option_ind

    if video_llm_pred is None:
        return "fail"
    else:
        return 1 if video_llm_pred == answer or video_llm_pred == answer.split(":")[0] or video_llm_pred == answer.split(": ")[1] or video_llm_pred == answer.split(": ")[0].split()[1] else 0  # noqa


def eval_rule_multi_choice(line):
    if line['prediction'] == line['answer']:
        return 1
    elif line['prediction'] in ['A', 'B', 'C', 'D']:
        return 1 if line['prediction'] == line['answer'][0] else 0
    elif any(line['prediction'].startswith(prefix) for prefix in ['A.', 'B.', 'C.', 'D.']):
        return 1 if line['prediction'].split('.')[0] == line['answer'][0] else 0
    elif any(line['prediction'].startswith(prefix) for prefix in ['A)', 'B)', 'C)', 'D)']):
        return 1 if line['prediction'].split(')')[0] == line['answer'][0] else 0
    else:
        return "fail"


def eval_rule_YorN(video_llm_output):
    # Extract the yes/no predction from the original video llm output
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "yes"
    elif video_llm_output.startswith("no"):
        return "no"
    else:
        return False


def llm_output_to_rating(llm_output):
    if not ('Correct' in llm_output or 'Incorrect' in llm_output):
        print(f"Warning: LLM output is not in the correct format: {llm_output}")
        rating = 0
        return rating
    if llm_output.startswith('Correct'):
        rating = 1
    elif llm_output.startswith('Incorrect'):
        rating = 0
    elif ('Correct' in llm_output) and ('Incorrect' not in llm_output):
        rating = 1
    elif 'Incorrect' in llm_output:
        rating = 0
    return rating


def parse_llm_output(llm_output, gt_answer):
    if llm_output == "invalid_request_error" or not llm_output:
        eval_result = {"rating": -1, "chatgpt-answer": None, "chatgpt-reasoning": None}
        return eval_result

    eval_result = {}
    lines = llm_output.split("\n")

    for line in lines:
        line = line.strip()
        if "Reasoning" in line:
            eval_result['chatgpt-reasoning'] = line.replace("Reasoning:", "").strip()
        if "Answer" in line:
            eval_result['chatgpt-answer'] = line.replace("Answer:", "").strip()

    if "chatgpt-answer" not in eval_result:
        eval_result['chatgpt-answer'] = llm_output
    if "chatgpt-reasoning" not in eval_result:
        eval_result['chatgpt-reasoning'] = None

    # Check if the chatgpt answer is the ground-truth answer
    # calculate the number of 'A.', 'B.', 'C.', 'D.' in chatgpt-answer
    answer_counts = sum(eval_result['chatgpt-answer'].count(prefix) for prefix in ['A.', 'B.', 'C.', 'D.'])  # noqa
    if eval_result['chatgpt-answer'].split(". ")[0] == gt_answer.split(". ")[0] and answer_counts == 1:
        eval_result['rating'] = 1
    else:
        eval_result['rating'] = 0
    return eval_result


def evaluate_tempcompass_mcq(model, line):
    eval_rules_dict = {
        'caption_matching': eval_rule_caption_matching,
        'multi-choice': eval_rule_multi_choice
    }
    gpt_eval_prompt = {
        'multi-choice': '{}\nMulti-Choice Question:\n{}\nGround-Truth Answer: {}\nModel Prediction: {}',
        'caption_matching': '{}\nCaption Matching Question:\n{}\nGround-Truth Answer: {}\nModel Prediction: {}'
    }
    base_prompt = {
        'multi-choice': system_prompt_multi_choice,
        'caption_matching': system_prompt_caption_matching
    }
    eval_result = {
        "question": line['question'],
        "answer": line['answer'],
        "prediction": line['prediction'],
        "task_type": line['task_type'],
        "candidates": line['candidates'],
        "match_success": True
    }
    result = eval_rules_dict[line['task_type']](line)
    if result == "fail":
        eval_result['match_success'] = False
        if model is None:
            eval_result['rating'] = 0
        else:
            prompt_template = gpt_eval_prompt[line['task_type']]
            prompt = prompt_template.format(base_prompt[line['task_type']], line['question'], line['answer'], line['prediction'])  # noqa
            llm_output = model.generate(prompt)
            result = llm_output_to_rating(llm_output)
            eval_result['chatgpt-response'] = llm_output
            eval_result['rating'] = result
    else:
        eval_result['rating'] = result

    return eval_result


def evaluate_tempcompass_captioning(model, line):
    prompt = (
        f"{system_prompt_captioning}\n"
        f"Video Description:{line['prediction']}\n"
        f"Multi-Choice Question:\n{line['mc_question']}\n"
    )
    if model is not None:
        llm_output = model.generate(prompt)
        eval_result = parse_llm_output(llm_output, gt_answer=line['mc_answer'])
        return eval_result
    else:
        raise ValueError("Model is None, TempCompass Captioning task not supported exact matching")  # noqa


def evaluate_tempcompass_YorN(model, line):
    prompt = (
        f"{system_prompt_YorN}\n"
        f"Yes/No Question:\n{line['question']}\n"
        f"Ground-Truth Answer: {line['answer']}\n"
        f"Model Prediction: {line['prediction']}"
    )
    result = eval_rule_YorN(line['prediction'])
    eval_result = {
        "question": line['question'],
        "answer": line['answer'],
        "prediction": line['prediction'],
        "match_success": True
    }
    if result:
        eval_result['rating'] = 1 if result == line['answer'] else 0
    elif model is None:
        eval_result['match_success'] = False
        eval_result['rating'] = 0
    else:
        eval_result['match_success'] = False
        llm_output = model.generate(prompt)
        result = llm_output_to_rating(llm_output)
        eval_result['chatgpt-response'] = llm_output
        eval_result['rating'] = result
    return eval_result


def get_dimension_rating(score_file):
    data = load(score_file)
    result_dict = {}
    for idx, item in data.iterrows():
        dict_key = item['dim'] + '. ' + item['task_type']
        if dict_key not in result_dict:
            result_dict[dict_key] = [0,0]
        result_dict[dict_key][0] += int(item['score'])
        result_dict[dict_key][1] += 1
    return result_dict
