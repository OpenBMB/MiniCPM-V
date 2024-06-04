## Using MiniCPM-Llama3-V-2_5 with Multiple GPUs

Due to the limited memory capacity of a single GPU, it may be impossible to load the entire MiniCPMV model (the model weights account for 18 GiB) onto one device for inference (assume one gpu only has 12 GiB or 16 GiB GPU memory). To address this limitation, multi-GPU inference can be employed, where the model's layers are distributed across multiple GPUs.

A minimal modification method can be used to achieve this distribution, ensuring that the layers are assigned to different GPUs with minimal changes to the original model structure.

To implement this, we utilize features provided by `accelerate` library. 

Install all requirements of MiniCPM-Llama3-V-2_5, additionally, you also need to install `accelerate`.

```bash
pip install accelerate
```

<br/>

### Example Usage for `2x16GiB` GPUs

Now we consider a demo for two GPUs, where each GPU has 16 GiB GPU memory.

1. Import necessary libraries.

```python
from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
```

2. Download model weights.

```python
MODEL_PATH = '/local/path/to/MiniCPM-Llama3-V-2_5' # you can download in advance or use `openbmb/MiniCPM-Llama3-V-2_5`
```

3. Determine the distribution of layers on multiple GPUs. 

```python
max_memory_each_gpu = '10GiB' # Define the maximum memory to use on each gpu, here we suggest using a balanced value, because the weight is not everything, the intermediate activation value also uses GPU memory (10GiB < 16GiB)

gpu_device_ids = [0, 1] # Define which gpu to use (now we have two GPUs, each has 16GiB memory)

no_split_module_classes = ["LlamaDecoderLayer"]

max_memory = {
    device_id: max_memory_each_gpu for device_id in gpu_device_ids
}

config = AutoConfig.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
)

with init_empty_weights():
    model = AutoModel.from_config(
        config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory, no_split_module_classes=no_split_module_classes
)

print("auto determined device_map", device_map)

# Here we want to make sure the input and output layer are all on the first gpu to avoid any modifications to original inference script.

device_map["llm.model.embed_tokens"] = 0
device_map["llm.model.layers.0"] = 0
device_map["llm.lm_head"] = 0
device_map["vpm"] = 0
device_map["resampler"] = 0

print("modified device_map", device_map)

```

You may see this output:

```
modified device_map OrderedDict([('llm.model.embed_tokens', 0), ('llm.model.layers.0', 0), ('llm.model.layers.1', 0), ('llm.model.layers.2', 0), ('llm.model.layers.3', 0), ('llm.model.layers.4', 0), ('llm.model.layers.5', 0), ('llm.model.layers.6', 0), ('llm.model.layers.7', 0), ('llm.model.layers.8', 0), ('llm.model.layers.9', 0), ('llm.model.layers.10', 0), ('llm.model.layers.11', 0), ('llm.model.layers.12', 0), ('llm.model.layers.13', 0), ('llm.model.layers.14', 0), ('llm.model.layers.15', 0), ('llm.model.layers.16', 1), ('llm.model.layers.17', 1), ('llm.model.layers.18', 1), ('llm.model.layers.19', 1), ('llm.model.layers.20', 1), ('llm.model.layers.21', 1), ('llm.model.layers.22', 1), ('llm.model.layers.23', 1), ('llm.model.layers.24', 1), ('llm.model.layers.25', 1), ('llm.model.layers.26', 1), ('llm.model.layers.27', 1), ('llm.model.layers.28', 1), ('llm.model.layers.29', 1), ('llm.model.layers.30', 1), ('llm.model.layers.31', 1), ('llm.model.norm', 1), ('llm.lm_head', 0), ('vpm', 0), ('resampler', 0)])
```

4. Next, use the `device_map` to dispatch the model layers to corresponding gpus.

```python
load_checkpoint_in_model(
    model, 
    MODEL_PATH, 
    device_map=device_map)

model = dispatch_model(
    model, 
    device_map=device_map
)

torch.set_grad_enabled(False)

model.eval()
```



5. Chat!

```python
image_path = '/local/path/to/test.png'

response = model.chat(
    image=Image.open(image_path).convert("RGB"),
    msgs=[
        {
            "role": "user",
            "content": "guess what I am doing?"
        }
    ],
    tokenizer=tokenizer
)

print(response)
```

In this case the OOM (CUDA out of memory) problem may be eliminated. We have tested that:

- it works well for `3000` text input tokens and `1000` text output tokens.
- it works well for a high-resolution input image.

<br/>

### Usage for general cases

It is similar to the previous example, but you may consider modifying these two variables.

```python
max_memory_each_gpu = '10GiB' # Define the maximum memory to use on each gpu, here we suggest using a balanced value, because the weight is not everything, the intermediate activation value also uses GPU memory (10GiB < 16GiB)

gpu_device_ids = [0, 1, ...] # Define which gpu to use (now we have two GPUs, each has 16GiB memory)
```

You may use the following shell script to monitor the memory usage during inference, if there is OOM, try to reduce `max_memory_each_gpu`, to make memory pressure more balanced across all gpus.

```bash
watch -n1 nvidia-smi
```

<br/>


### References

[Ref 1](https://zhuanlan.zhihu.com/p/639850033)

