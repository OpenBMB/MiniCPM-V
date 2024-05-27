import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..smp import *
from ..utils import DATASET_TYPE
from .base import BaseModel


class MiniCPM_Llama3_V(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.ckpt = model_path

        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        if '.pt' in model_path:
            print(f'load from {model_path}')
            self.state_dict = torch.load(self.ckpt, map_location='cpu')
            self.model.load_state_dict(self.state_dict, strict=False)
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3
        self.options_system_prompt = ('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'
        

    def use_custom_prompt(self, dataset):
        if listinstr(['multi-choice', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'multi-choice':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt +  "\nPlease just indicate your choice."
            else:
                system_prompt = self.wo_options_system_prompt

            if 'MMMU' in dataset: # Corner Case
                prompt = system_prompt + '\n' + prompt
                system_prompt = ''
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + " Yes or No?"
            prompt = question
        
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench'], dataset):
                system_prompt = ""
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None):
        if DATASET_TYPE(dataset) == 'multi-choice':
            max_new_tokens = 200
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3
        else:
            max_new_tokens = 1024

        '''
        nums_beams = 3
        '''
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []
        
    #     message = [
    #     {'type': 'text', 'value': 'sys prompt'},
    #     {'type': 'image', 'value': '/path/to/image1.jpg'},
    #     {'type': 'text', 'value': 'Here is an image:'},
    # ]

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image = None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        # print(f"content: {content}, res: {res}")
        return res
