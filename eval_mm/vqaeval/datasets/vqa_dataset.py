import json
import os
import re
from torch.utils.data import Dataset

def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()
    
class textVQADataset(Dataset):
    def __init__(
        self,
        image_dir="./downloads/TextVQA/train_images",
        ann_path="./downloads/TextVQA/TextVQA_0.5.1_val.json",
    ):
        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_id = self.data[idx]['image_id']
        qid = self.data[idx]['question_id']
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        item = {
            "question_id": qid,
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        
        return item
    
class docVQADataset(Dataset):
    def __init__(
        self,
        image_dir= "./downloads/DocVQA/spdocvqa_images",
        ann_path= "./downloads/DocVQA/val_v1.0_withQT.json",
        ocr_token_path=None
    ):

        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir = image_dir
        self.ann_path = ann_path
        if ocr_token_path:
            self.ocr_token_data = {item['image_id']: item for item in json.load(open(ocr_token_path, "r"))["data"]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_id = self.data[idx]['questionId']  
        relative_img_path = self.data[idx]['image']
        corrected_relative_img_path = relative_img_path.replace("documents", "images")
        img_path = os.path.join(self.image_dir, corrected_relative_img_path)
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        
        question_type = self.data[idx]['question_types']
        
        return {
            "question_id": question_id,  
            "image_path": img_path,
            "question": question,
            "gt_answers": answers,
            'question_type': question_type,
        }


class docVQATESTDataset(Dataset):
    def __init__(
        self,
        image_dir= "./downloads/DocVQA/spdocvqa_images",
        ann_path= "./downloads/DocVQA/test_v1.0.json",
        ocr_token_path=None
    ):

        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir = image_dir
        self.ann_path = ann_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_id = self.data[idx]['questionId']  
        relative_img_path = self.data[idx]['image']
        corrected_relative_img_path = relative_img_path.replace("documents", "images")
        img_path = os.path.join(self.image_dir, corrected_relative_img_path)
        question = self.data[idx]['question']
        
        
        return {
            "question_id": question_id,  
            "image_path": img_path,
            "question": question,
            "gt_answers": "",
            'question_type': "",
        }
