<div align="center">

# OmniLMM
**Large multi-modal models for strong performance and efficient deployment**
<!-- <p align="center">
  <a href="#-viscpm-chat">Multimodal Conversation Model VisCPM-Chat</a> •
  <a href="#-viscpm-paint">Text-to-image Model VisCPM-Paint</a> •
  <a href="#-inference">Inference</a> •
  <a href="https://arxiv.org/pdf/2308.12038.pdf">Paper</a>
</p> -->

</div>


OmniLMM is a family of open-source large multimodal models (LMMs) that are adept at vision & language modeling. The model accepts images and text input, and emitts text outputs. We release two versions of OmniLMM that are targeted at strong performance and efficient deployment.
- OmniLMM 12B, the most capable version that achives state-of-the-art performance among open-source models with comparable sizes on MMMU.
- OmniLMM 3B, the efficient version that can be deployed on edge devices with promising performance.

## OmniLMM 12B
OmniLMM 12B is the most capable version. The model is built based on EVA-E 5B and Zephyr 7B, and trained on mulitmodal data in a curriculum fashion. It has three notable features:

- **Strong Performance.** OmniLMM 12B achieves state-of-the-art performance on MMMU among open-source models with comparable sizes, surpassing established open-source LMMs on mulitple benchmarks (including MME, MMBench and SEED-Bench, etc). The model also supports OCR capability and endows rich multimodal world knowledge.

- **Trustworthy Behavior.** LMMs are known for suffering from hallucination, often generating text that is not factually grounded in images (e.g., faithfully describing non-existing objects in images). OmniLMM 12B is first state-of-the-art open-source LMM aligned via multimodal RLHF for trustworthy behavior, and ranked #1 among open-source models on MMHalBench and Object Halbench.

- **Realtime Mulitmodal Interaction.** We combine the OmniLMM 12B and ChatGPT3.5 into a realtime multimodal interactive assistant. The assistant accepts video stream from camera and speech stream from microphone, and emitts speech output. While still primary, we find the model can replicate some of the fun cases shown in Gemini Demo video, without any video edition.

TODO：实验结果，可以放个表格截图（基准：MMMU、MME、MMBench、SEED-Bench、MMHALBench、Object Halbench；模型：Qwen-VL-Chat、CogVLM、GPT-4V、Gemini等） @余天予

TODO：case画图展示 @蔡天驰

## OmniLMM 3B
OmniLMM 3B (i.e., MiniCPM-Omni) is an efficient version for deployment. The model is built based on SigLip 400M and MiniCPM 2.4B, and trained in a smilar way to OmniLMM 12B. Notable features include:

- **High Efficiency.** OmniLLM 3B can be efficiently deployed on most GPU cards and personal computers, and even on edge devices such as mobilephones. Due to the significantly fewer tokens used to represent the images (i.e., 64 in OmniLMM 3B vs. 512+ in counterpart models), OmniLMM 3B can operate with less memory cost and higher speed during inference.

- **Promising Performance.** OmniLMM 3B achieves promising performance on multiple benchmarks (including MMMU, MME and MMbech), surpassing existing LMMs built on Phi-2. The model supports bilingual mulitmodal interaction in English and Chinese, and endows rich multimodal world knowledge.


| **Method**       | **Parameters** | **MME(P)** | **MMB-dev(en)** | **MMB-dev(zh)** | **CCBench** | **MMMU-val** | **CMMMU-val** |
|:------------:|:-------:|:----------:|:---------------:|:---------------:|:-----------:|:------------:|:-------------:|
| LLaVA-Phi    | 3B      | 1335       | 59.8            | -               |             | -            | -             |
| MobileVLM    | 3B      | 1289       | 59.6            | -               |             | -            | -             |
| Imp-v1       | 3B      | 1434       | 66.5            | -               |             | -            | -             |
| Qwen-VL-Chat | 9.6B    | 1487       | 60.6            | 56.7            | 41.2        | 35.9         | 30.7          |
| MiniCPM-Omni | 3B      | 1452       | 67.3            | 61.9            | 37.8        | 34.7         | 32.1          |

TODO：视频展示手机端效果？ @蔡天驰

## Get Started

## ⚙️ Install

1. Clone this repository and navigate to source folder

```bash
git clone https://github.com/OpenBMB/OmniLMM.git
cd OmniLMM
```

2. Create conda environment

```Shell
conda create -n OmniLMM python=3.10 -y
conda activate OmniLMM
```

3. Install dependencies

```shell
pip install -r requirements.txt
```

## 💡 Inference

### Model Zoo
| Model                | Description       | Download Link |
|----------------------|-------------------|---------------|
| OmniLMM-12B | OmniLMM 12B is the most capable version                                 | [download](https://huggingface.co/openbmb/OmniLMM-12B/blob/main/pytorch_model.v1.bin) |
| OmniLMM-3B  | OmniLMM 3B (i.e., MiniCPM-Omni) is an efficient version for deployment. | [download](https://huggingface.co/openbmb/OmniLMM-3B/blob/main/pytorch_model.v1.bin)  |

### OmniLMM-12B
After downloading the checkpoints, please refer to the following codes to run `OmniLMM` (replace `'/path/to/checkpoint'` with actually path of downloaded checkpoint).

#### Multi-turn Conversation

<div align="center">
<img src="data/COCO_test2015_000000262144.jpg" width="660px">
</div>

```python
from chat import OmniLMMChat, img2base64

# Load and initialize the model
model_path = '/path/to/checkpoint'
chat_model = OmniLMMChat(model_path)

# We perform security checks on the input images by default.
im_64 = img2base64('./data/COCO_test2015_000000262144.jpg')

# First round chat 
msgs = [{"role": "user", "content": "What are the people doing?"}]
input = {
    "image": im_64,
    "question": json.dumps(msgs, ensure_ascii=True)
}
answer = chat_model.process(input)
print(answer)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": answer})
msgs.append({"role": "user", "content": "Describe the image"})
input = {
    "image": im_64,
    "question": json.dumps(msgs, ensure_ascii=True)
}
answer = chat_model.process(input)
print(answer)
```

We can obtain the following results:
```
"The people in the image are playing baseball. One person is pitching a ball, another one is swinging a bat to hit it, and there's also an umpire present who appears to be watching the game closely."

"The image depicts a baseball game in progress. A pitcher is throwing the ball, while another player is swinging his bat to hit it. An umpire can be seen observing the play closely."
```

TODO：使用文档（安装、使用、提供Demo入口，包括3B和12B） @朱宏吉

## 🏫 Institutions

This project is developed by the following institutions:

- <img src="figures/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="figures/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)
- <img src="figures/zhihu.webp" width="28px"> [Zhihu](https://www.zhihu.com/ )


