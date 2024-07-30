## SWIFT install
You can quickly install SWIFT using bash commands.

``` bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## SWIFT Infer
Inference using SWIFT can be carried out in two ways: through a command line interface and via Python code.

### Quick start
Here are the steps to launch SWIFT from the Bash command line:

1. Run the bash code will download the model of MiniCPM-Llama3-V-2_5 and run the inference
``` shell
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2_5-chat
```

2. You can also run the code with more arguments below to run the inference:
``` 
model_id_or_path  # Can be the model ID from Hugging Face or the local path to the model
infer_backend ['AUTO', 'vllm', 'pt']  # Backend for inference, default is auto
dtype ['bf16', 'fp16', 'fp32', 'AUTO']  # Computational precision
max_length  # Maximum length
max_new_tokens: int = 2048  # Maximum number of tokens to generate
do_sample: bool = True  # Whether to sample during generation
temperature: float = 0.3  # Temperature coefficient during generation
top_k: int = 20 
top_p: float = 0.7
repetition_penalty: float = 1.  # Penalty for repetition
num_beams: int = 1  # Number of beams for beam search
stop_words: List[str] = None  # List of stop words
quant_method ['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm']  # Quantization method for the model
quantization_bit [0, 1, 2, 3, 4, 8]  # Default is 0, which means no quantization is used
```
3. Example:
``` shell
CUDA_VISIBLE_DEVICES=0，1 swift infer \
--model_type minicpm-v-v2_5-chat \
--model_id_or_path /root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5 \
--dtype bf16 
```
### Python code with SWIFT infer
The following demonstrates using Python code to initiate inference with the MiniCPM-Llama3-V-2_5 model through SWIFT.

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Set the number of GPUs to use

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)  # Import necessary modules

from swift.utils import seed_everything  # Set random seed
import torch

model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)  # Obtain the template type, primarily used for constructing special tokens and image processing workflow
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_id_or_path='/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5',
                                       model_kwargs={'device_map': 'auto'})  # Load the model, set model type, model path, model parameters, device allocation, etc., computation precision, etc.
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)  # Construct the template based on the template type
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']  # Image URL
query = '距离各城市多远？'  # Note: Query is still in Chinese, consider translating if needed
response, history = inference(model, template, query, images=images)  # Obtain results through inference
print(f'query: {query}')
print(f'response: {response}')

# Streaming output
query = '距离最远的城市是哪？'  # Note: Query is still in Chinese, consider translating if needed
gen = inference_stream(model, template, query, history, images=images)  # Call the streaming output interface
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
```

## SWIFT train
SWIFT supports training on the local dataset,the training steps are as follows:
1. Make the train data like this:
```jsonl
{"query": "What does this picture describe?", "response": "This picture has a giant panda.", "images": ["local_image_path"]}
{"query": "What does this picture describe?", "response": "This picture has a giant panda.", "history": [], "images": ["image_path"]}
{"query": "Is bamboo tasty?", "response": "It seems pretty tasty judging by the panda's expression.", "history": [["What's in this picture?", "There's a giant panda in this picture."], ["What is the panda doing?", "Eating bamboo."]], "images": ["image_url"]}
```
2. LoRA Tuning:

The LoRA target model are k and v weight in LLM you should pay attention to the eval_steps,maybe you should set the eval_steps to a large value, like 200000,beacuase in the eval time , SWIFT will return a memory bug so you should set the eval_steps to a very large value.
```shell
# Experimental environment: A100
# 32GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
--model_type minicpm-v-v2_5-chat \
--dataset coco-en-2-mini \
```
3. All parameters finetune:

When the argument of lora_target_modules is ALL, the model will finetune all the parameters.
```shell
CUDA_VISIBLE_DEVICES=0,1 swift sft \
--model_type minicpm-v-v2_5-chat \
--dataset coco-en-2-mini \
--lora_target_modules ALL \
--eval_steps 200000
```

## LoRA Merge and Infer
The LoRA weight can be merge to the base model and then load to infer.

1. Load the LoRA weight to infer run the follow code:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
--ckpt_dir /your/lora/save/checkpoint
```
2. Merge the LoRA weight to the base model:

The code will load and merge the LoRA weight to the base model, save the merge model to the LoRA save path and load the merge model to infer
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
--ckpt_dir your/lora/save/checkpoint \
--merge_lora true
```