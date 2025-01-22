# Best Practice with LLaMA-Factory

## Contents <!-- omit in toc -->

- [Support Models](#Support-Models)
- [LLaMA-Factory Installation](#LLaMA-Factory-Installation)
- [Dataset Prepare](#Dataset-Prepare)
  - [Image Dataset](#Image-Dataset)
  - [Video Dataset](#Video-Dataset)
  - [Audio Dataset](#Audio-Dataset)
- [Lora Fine-Tuning](#Lora-Fine-Tuning)
- [Full Parameters Fine-Tuning](#Full-Parameters-Fine-Tuning)
- [Inference](#Inference)

## Support Models
* [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)
* [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)

## LLaMA-Factory Installation

You can install LLaMA-Factory using commands below.

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,minicpm_v]"
mkdir configs # let's put all yaml files here 
```

## Dataset Prepare

Refer to [data/dataset_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json) to add your customised dataset. Let's use the two existing demo datasets `mllm_demo` and `mllm_video_demo` as examples.

### Image Dataset

Refer to image sft demo data: [data/mllm_demo.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_demo.json)

<details>
  <summary>
    <b>data/mllm_demo.json</b>
  </summary>

```json
[
  {
    "messages": [
      {
        "content": "<image>Who are they?",
        "role": "user"
      },
      {
        "content": "They're Kane and Gretzka from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "What are they doing?",
        "role": "user"
      },
      {
        "content": "They are celebrating on the soccer field.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg"
    ]
  },
  {
    "messages": [
      {
        "content": "<image>Who is he?",
        "role": "user"
      },
      {
        "content": "He's Thomas Muller from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "Why is he on the ground?",
        "role": "user"
      },
      {
        "content": "Because he's sliding on his knees to celebrate.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/2.jpg"
    ]
  },
  {
    "messages": [
      {
        "content": "<image>Please describe this image",
        "role": "user"
      },
      {
        "content": "Chinese astronaut Gui Haichao is giving a speech.",
        "role": "assistant"
      },
      {
        "content": "What has he accomplished?",
        "role": "user"
      },
      {
        "content": "He was appointed to be a payload specialist on Shenzhou 16 mission in June 2022, thus becoming the first Chinese civilian of Group 3 in space on 30 May 2023. He is responsible for the on-orbit operation of space science experimental payloads.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/3.jpg"
    ]
  }
]
```

</details>


### Video Dataset

Refer to video sft demo data: [data/mllm_video_demo.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_video_demo.json)

<details>
  <summary>
    <b>data/mllm_video_demo.json</b>
  </summary>

```json
[
  {
    "messages": [
      {
        "content": "<video>Why is this video funny?",
        "role": "user"
      },
      {
        "content": "Because a baby is reading, and he is so cute!",
        "role": "assistant"
      }
    ],
    "videos": [
      "mllm_demo_data/1.mp4"
    ]
  },
  {
    "messages": [
      {
        "content": "<video>What is she doing?",
        "role": "user"
      },
      {
        "content": "She is cooking.",
        "role": "assistant"
      }
    ],
    "videos": [
      "mllm_demo_data/2.avi"
    ]
  },
  {
    "messages": [
      {
        "content": "<video>What's in the video?",
        "role": "user"
      },
      {
        "content": "A baby is playing in the living room.",
        "role": "assistant"
      }
    ],
    "videos": [
      "mllm_demo_data/3.mp4"
    ]
  }
]
```

</details>

### Audio Dataset

Refer to audio sft demo data: [data/mllm_audio_demo.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_audio_demo.json)

<details>
  <summary>
    <b>data/mllm_audio_demo.json</b>
  </summary>

```json
[
  {
    "messages": [
      {
        "content": "<audio>What's that sound?",
        "role": "user"
      },
      {
        "content": "It is the sound of glass shattering.",
        "role": "assistant"
      }
    ],
    "audios": [
      "mllm_demo_data/1.mp3"
    ]
  },
  {
    "messages": [
      {
        "content": "<audio>What can you hear?",
        "role": "user"
      },
      {
        "content": "A woman is coughing.",
        "role": "assistant"
      }
    ],
    "audios": [
      "mllm_demo_data/2.wav"
    ]
  },
  {
    "messages": [
      {
        "content": "<audio>What does the person say?",
        "role": "user"
      },
      {
        "content": "Mister Quiller is the apostle of the middle classes and we are glad to welcome his gospel.",
        "role": "assistant"
      }
    ],
    "audios": [
      "mllm_demo_data/3.flac"
    ]
  }
]
```

</details>

## Lora Fine-Tuning

We can use one command to do lora sft:

```shell
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/minicpmo_2_6_lora_sft.yaml
```

<details>
  <summary>
    <b>configs/minicpmo_2_6_lora_sft.yaml</b>
  </summary>

```yaml
### model
model_name_or_path: openbmb/MiniCPM-o-2_6 # MiniCPM-o-2_6 MiniCPM-V-2_6
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj

### dataset
dataset: mllm_demo # mllm_demo mllm_video_demo mllm_audio_demo
template: minicpm_v
cutoff_len: 3072
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/minicpmo_2_6/lora/sft
logging_steps: 1
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_total_limit: 10

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 20.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
save_only_model: true

### eval
do_eval: false
```

</details>

### Lora Model Export

One command to export lora model

```shell
llamafactory-cli export configs/minicpmo_2_6_lora_export.yaml
```

<details>
  <summary>
    <b>configs/minicpmo_2_6_lora_export.yaml</b>
  </summary>

```yaml
### model
model_name_or_path: openbmb/MiniCPM-o-2_6 # MiniCPM-o-2_6 MiniCPM-V-2_6
adapter_name_or_path: saves/minicpmo_2_6/lora/sft
template: minicpm_v
finetuning_type: lora
trust_remote_code: true

### export
export_dir: models/minicpmo_2_6_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

</details>

## Full Parameters Fine-Tuning

We can use one command to do full sft:

```shell
llamafactory-cli train configs/minicpmo_2_6_full_sft.yaml
```

<details>
  <summary>
    <b>configs/minicpmo_2_6_full_sft.yaml</b>
  </summary>

```yaml
### model
model_name_or_path: openbmb/MiniCPM-o-2_6 # MiniCPM-o-2_6 MiniCPM-V-2_6
trust_remote_code: true
freeze_vision_tower: true
print_param_status: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: configs/deepspeed/ds_z2_config.json

### dataset
dataset: mllm_demo # mllm_demo mllm_video_demo
template: minicpm_v
cutoff_len: 3072
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/minicpmo_2_6/full/sft
logging_steps: 1
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_total_limit: 10

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 20.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
save_only_model: true

### eval
do_eval: false
```
</details>

## Inference

### Web UI ChatBox

Refer [LLaMA-Factory doc](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples#inferring-lora-fine-tuned-models) for more inference usages.

For example, we can use one command to run web chat:

```shell
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat configs/minicpmo_2_6_infer.yaml
```

<details>
  <summary>
    <b>configs/minicpmo_2_6_infer.yaml</b>
  </summary>

```yaml
model_name_or_path: saves/minicpmo_2_6/full/sft
template: minicpm_v
infer_backend: huggingface
trust_remote_code: true
```
</details>

### Official Code
You can also use official code to inference

<details>
  <summary>
    <b>official inference code</b>
  </summary>
  
```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model_id = "saves/minicpmo_2_6/full/sft"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

image = Image.open('data/mllm_demo_data/1.jpg').convert('RGB')
question = 'Who are they??'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(res)
```

</details>