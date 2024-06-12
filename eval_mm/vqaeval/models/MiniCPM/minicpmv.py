
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

Image.MAX_IMAGE_PIXELS = 1000000000

max_token  = {
    'docVQA': 100,
    'textVQA': 100,
    "docVQATest": 100
}

class MiniCPM_V:

    def __init__(self, model_path, ckpt, device=None)->None:
        self.model_path = model_path
        self.ckpt = ckpt
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).eval()
        if self.ckpt is not None:
            self.ckpt = ckpt
            self.state_dict = torch.load(self.ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(self.state_dict)
            
        self.model = self.model.to(dtype=torch.float16)
        self.model.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

    def generate(self, images, questions, datasetname):
        image = Image.open(images[0]).convert('RGB')
        try:
            max_new_tokens = max_token[datasetname]
        except:
            max_new_tokens = 1024
        if (datasetname == 'docVQA') or (datasetname == "docVQATest") :
            prompt = "Answer the question directly with single word." + "\n" + questions[0]
        elif (datasetname == 'textVQA') :
            prompt = "Answer the question directly with single word." + '\n'+ questions[0]
        
        msgs = [{'role': 'user', 'content': prompt}]
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=3
        )
        res = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        
        return [res]
    
    def generate_with_interleaved(self, images, questions, datasetname):
        try:
            max_new_tokens = max_token[datasetname]
        except:
            max_new_tokens = 1024
        
        prompt = "Answer the question directly with single word."
        
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=3
        )
        
        content = []
        message = [
            {'type': 'text', 'value': prompt},
            {'type': 'image', 'value': images[0]},
            {'type': 'text', 'value': questions[0]}
        ]
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        print(f"Q: {content}, \nA: {res}")
        return [res]
