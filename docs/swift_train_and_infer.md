## Swift install
You can quickly install Swift using bash commands.

``` bash
    git clone https://github.com/modelscope/swift.git
    cd swift
    pip install -r requirements.txt
    pip install -e '.[llm]'
```

## Swift Infer
Inference using Swift can be carried out in two ways: through a command line interface and via Python code.

### Quick start
Here are the steps to launch Swift from the Bash command line:

1. Run the bash code will download the model of MiniCPM-Llama3-V-2_5 and run the inference
``` shell
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2_5-chat
```

2. You can also run the code with more arguments below to run the inference:
``` 
    model_id_or_path # 可以写huggingface的模型id或者本地模型地址
    infer_backend ['AUTO', 'vllm', 'pt'] # 后段推理，默认auto
    dtype ['bf16', 'fp16', 'fp32', 'AUTO'] # 计算精度
    max_length # 最大长度
    max_new_tokens: int = 2048 #最多生成多少token
    do_sample: bool = True # 是否采样
    temperature: float = 0.3 # 生成时的温度系数
    top_k: int = 20 
    top_p: float = 0.7
    repetition_penalty: float = 1.
    num_beams: int = 1
    stop_words: List[str] = None
    quant_method ['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm'] # 模型的量化方式
    quantization_bit [0, 1, 2, 3, 4, 8] 默认是0，代表不使用量化
```
3. Example:
``` shell
    CUDA_VISIBLE_DEVICES=0，1 swift infer \
    --model_type minicpm-v-v2_5-chat \
    --model_id_or_path /root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5 \
    --dtype bf16 
```
### Python code with swift infer
The following demonstrates using Python code to initiate inference with the MiniCPM-Llama3-V-2_5 model through Swift.

```python
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 设置显卡数

    from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType,
        get_default_template_type, inference_stream
    ) # 导入必要模块

    from swift.utils import seed_everything # 设置随机种子
    import torch

    model_type = ModelType.minicpm_v_v2_5_chat
    template_type = get_default_template_type(model_type) # 获取模板类型，主要是用于特殊token的构造和图像的处理流程
    print(f'template_type: {template_type}')

    model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                        model_id_or_path='/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5',
                                        model_kwargs={'device_map': 'auto'}) # 加载模型，并设置模型类型，模型路径，模型参数，设备分配等，计算精度等等
    model.generation_config.max_new_tokens = 256
    template = get_template(template_type, tokenizer) # 根据模版类型构造模板
    seed_everything(42)

    images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png'] # 图片地址
    query = '距离各城市多远？'
    response, history = inference(model, template, query, images=images) # 推理获得结果
    print(f'query: {query}')
    print(f'response: {response}')

    # 流式
    query = '距离最远的城市是哪？'
    gen = inference_stream(model, template, query, history, images=images) # 调用流式输出接口
    print_idx = 0
    print(f'query: {query}\nresponse: ', end='')
    for response, history in gen:
        delta = response[print_idx:]
        print(delta, end='', flush=True)
        print_idx = len(response)
    print()
    print(f'history: {history}')
```

## Swift train
Swift supports training on the local dataset,the training steps are as follows:
1. Make the train data like this:
```jsonl
{"query": "这张图片描述了什么", "response": "这张图片有一个大熊猫", "images": ["local_image_path"]}
{"query": "这张图片描述了什么", "response": "这张图片有一个大熊猫", "history": [], "images": ["image_path"]}
{"query": "竹子好吃么", "response": "看大熊猫的样子挺好吃呢", "history": [["这张图有什么", "这张图片有大熊猫"], ["大熊猫在干嘛", "吃竹子"]], "images": ["image_url"]}
```
2. Lora Tuning:

    The lora target model are k and v weight in llm you should pay attention to the eval_steps,maybe you should set the eval_steps to a large value, like 200000,beacuase in the eval time , swift will return a memory bug so you should set the eval_steps to a very large value.
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

## Lora Merge and Infer
The lora weight can be merge to the base model and then load to infer.

1. Load the lora weight to infer run the follow code:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer    \
 --ckpt_dir /your/lora/save/checkpoint
```
2. Merge the lora weight to the base model:

    The code will load and merge the lora weight to the base model, save the merge model to the lora save path and load the merge model to infer
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir your/lora/save/checkpoint \
    --merge_lora true
```