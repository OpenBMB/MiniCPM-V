import itertools
import json
import os
import re
from collections import namedtuple

import torch
from tqdm import tqdm


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
    
def collate_fn_vqa(batches):
    '''
    '''
    image_paths = [_['image_path'] for _ in batches]
    questions = [_['question'] for _ in batches]
    gt_answers = [_['gt_answers'] for _ in batches]
    ocr_tokens = [_['ocr_tokens'] if 'ocr_tokens' in _ else None for _ in batches]
    question_ids = [_['question_id'] if 'question_id' in _ else None for _ in batches]
    question_type = [_['question_type'] if 'question_type' in _ else None for _ in batches]

    return image_paths, questions, gt_answers, ocr_tokens, question_ids, question_type

def has_word(sentence, word):
    if word[0].isalnum():
        start_pattern = r"\b"
    else:
        start_pattern = r""

    if word[-1].isalnum():
        end_pattern = r"\b"
    else:
        end_pattern = r""

    pattern = start_pattern + re.escape(word) + end_pattern
    match = re.search(pattern, sentence)
    return bool(match)

def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
    def clean_text(self, text):
        text = text.replace("\n", " ").replace("\t", " ").strip()
        text = self.processPunctuation(text)
        text = self.processDigitArticle(text)
        return text
    
    def evaluate_vqa_human(self, answer, gt_answers):
        '''TextVQA, VQAv2, OKVQA, vizwiz'''
        answer = answer.replace("\n", " ").replace("\t", " ").strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        gt_answers = [self.processPunctuation(ans) for ans in gt_answers]
        gt_answers = [self.processDigitArticle(ans) for ans in gt_answers]

        gtAcc = [] 

        for idx, gtAnsDatum in enumerate(gt_answers):  
            otherGTAns = gt_answers[:idx] + gt_answers[idx+1:]

            matchingAns = [item for item in otherGTAns if answer == item]
            
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc) 

        avgGTAcc = float(sum(gtAcc)) / len(gtAcc) if gtAcc else 0  

        return avgGTAcc

    def evaluate_anls(self, answer, gt_answers, threshold=0.5):
        '''DOcVQA, InfographicsVQA, STVQA'''
        answer = ' '.join(answer.strip().lower().split())
        if not isinstance(gt_answers, list):
            gt_answers = [gt_answers]
        gt_answers = [' '.join(gt_answer.strip().lower().split()) for gt_answer in gt_answers]

        values = []
        for gt_answer in gt_answers:
            dist = levenshtein_distance(answer, gt_answer)
            length = max(len(answer), len(gt_answer))
            values.append(0.0 if length == 0 else float(dist) / float(length))

        score = 1 - min(values)
        
        score = 0 if score < threshold else score
        
        return score

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText


def evaluate_dataset(dataset_name, answer_file_path, model_name, method = None):
    with open(answer_file_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    eval = VQAEval()
    total_accuracy = 0
    num = 0
    Entry = namedtuple('Entry', ['text', 'bbox'])

    for item in predictions:
        gt_answers = item['gt_answers']
        answer = item['answer']
        if method is not None:
            pass
        if dataset_name in ["textVQA"]:
            if num == 0:
                print(f"evaluating vqa...")
            accuracy = eval.evaluate_vqa_human(answer, gt_answers)
        elif dataset_name in ['docVQA']:
            if  num == 0:
                print(f"evaluating anls...")
            accuracy = eval.evaluate_anls(answer, gt_answers)
        else:
            accuracy = eval.evaluate_has(answer, gt_answers)
        item['accuracy'] = accuracy

        total_accuracy += accuracy
        num += 1

    average_accuracy = total_accuracy / num
    print(f'{dataset_name}:{average_accuracy}')
    
    answer_model_method_path = answer_file_path.replace('.json', f'_{model_name}_{method}.json')
    with open(answer_model_method_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    return average_accuracy


def evaluate_VQA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    generate_method="interleave",
    answer_path='./answers',
):
    print(f"answer path:{answer_path}")

    sampler = None
    if torch.distributed.is_initialized():
        sampler=InferenceSampler(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_vqa
    )
    
    now_rank = torch.distributed.get_rank()
    
    answer_dir = os.path.join(answer_path, model_name, time)
    os.makedirs(answer_dir, exist_ok=True)
    
    image_list = []
    for item in dataset:
        image_list.append(item["image_path"])
    
    predictions = []
    
    for batch in tqdm(dataloader, desc="Running inference"):
        image_paths, questions, gt_answers, ocr_tokens_list, question_ids, question_type  = batch

        with torch.no_grad():
            if model_name != "minicpm":
                if model_name != "codellama":
                    outputs = model.generate(images=image_paths, questions=questions, datasetname=dataset_name)
                else:
                    outputs = model.generate()
            elif model_name == "minicpm":
                if generate_method == "old":
                    outputs = model.generate(images=image_paths, questions=questions, datasetname=dataset_name)
                elif generate_method == "interleave":
                    outputs = model.generate_with_interleaved(images=image_paths, questions=questions, datasetname=dataset_name)
                else:
                    raise Exception(f"Wrong generate paradigm {generate_method}!")
            
            for i in range(len(outputs)):
                answer_dict = {
                    'question_id': question_ids[i],
                    'question': questions[i],
                    'answer': outputs[i],
                    'gt_answers': gt_answers[i],
                    'image_path': image_paths[i],
                    'model_name': model_name,
                    'question_type': question_type[i]
                }              
                predictions.append(answer_dict)
                    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        merged_predictions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_predictions, predictions)
        predictions = [_ for _ in itertools.chain.from_iterable(merged_predictions)]

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    answer_file_path = os.path.join(answer_dir, f"{dataset_name}.json")
    print(f"answer_file_path:{answer_file_path}")
    
    with open(answer_file_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    if dataset_name in ["docVQATest"]:
        return -1.0

    return evaluate_dataset(answer_file_path=answer_file_path, dataset_name=dataset_name, model_name=model_name)
