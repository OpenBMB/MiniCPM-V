import json
import glob
import re

def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False
def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s

for model in glob.glob('./answer_save/*'):
    print(model, ':')
    result_list = sorted(glob.glob(f'{model}/*.json'))
    for task_result_path in result_list:
        taskname = task_result_path.split('/')[-1]
        taskname = taskname.split('.')[0]
        if taskname not in ['IIIT5K', 'svt', 'IC13_857', 'IC15_1811', 'svtp', 'ct80',
                            'cocotext', 'ctw', 'totaltext', 'HOST']:
            continue

        correct = 0
        num = 0
        with open(task_result_path, 'r') as f:
            dict = json.load(f)[:100]
            for i in range(len(dict)):
                gt_answers = dict[i]['gt_answers']
                answer = dict[i]['answer']
                gt_answers = remove_special_chars(gt_answers).lower()
                answer = remove_special_chars(answer).lower()
                if has_word(answer, gt_answers):
                    correct+=1
                num+=1
        print(f'{taskname:10s}:{float(correct)/num*100:.2f}')
    print('=' * 32)
