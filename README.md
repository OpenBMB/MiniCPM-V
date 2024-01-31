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
- OmniLMM 12B, the most capable version that achives leading performance among models with comparable sizes on multiple benchmarks.
- OmniLMM 3B, the efficient version that can be deployed on edge devices with promising performance.

## OmniLMM 12B
OmniLMM 12B is the most capable version. The model is built based on EVA-E 5B and Zephyr 7B, connected with a preceiver resampler layer, and trained on mulitmodal data in a curriculum fashion. The model has three notable features:

- **Strong Performance.** OmniLMM 12B achieves leading performance among models with comparable sizes, surpassing established LMMs on mulitple benchmarks (including MME, MMBench and SEED-Bench, etc). The model also supports OCR capability and endows rich multimodal world knowledge.

- **Trustworthy Behavior.** LMMs are known for suffering from hallucination, often generating text that is not factually grounded in images (e.g., faithfully describing non-existing objects in images). OmniLMM 12B is first state-of-the-art open-source LMM aligned via multimodal RLHF for trustworthy behavior (using our recent [RLHF-V](https://rlhf-v.github.io/) technique), and ranked #1 among open-source models on MMHalBench and Object Halbench.
  
- **Realtime Mulitmodal Interaction.** We combine the OmniLMM 12B and ChatGPT3.5 into a realtime multimodal interactive assistant. The assistant accepts video stream from camera and speech stream from microphone, and emitts speech output. While still primary, we find the model can replicate some of the fun cases shown in Gemini Demo video, without any video edition.

| **Method**       | Size | **MME(P)** | **MMMU val** | MMHal- Bench | SeedBench-I | LLaVA Bench W | MathVista | MMBench dev |
|:------------:|:-------:|:----------:|:---------------:|:---------------:|:------------:|:-------------:|--------------|--------------|
| GPT-4V | - | 1409 | 56.8 | 3.53 (70.8) | 71.6 | 93.1 | 47.8 | 75.1 |
| QWEN-VL-PLUS | - | 1681 | 45.2 | - | 65.7 | 73.7 | 36.0 | 66.2 |
| Qwen-VL-Chat | 9.6B | 1488   | 35.9         | 2.93 (59.4) | 64.8         | 67.7     | 33.8       | 60.6      |
| CogVLM | 17B | 1438 | 32.1 | 2.68 (52.1) | 68.8 | 73.9 | 34.7 | 63.7 |
| LLaVA 1.5 | 14B | 1531 | 36.4 | 2.71 (51.0) | 68.1 | 64.6 | 26.4 | 68.2 |
| Yi-VL | 6.7B | - | 39.1 | - | 66.1 | 39.9 | 28.0 | 68.2 |
| OmniLMM-12B | 12B    | 1637  | 40.7        | 3.45 (68.8) | 71.1        | 72.0      | 34.9      | 71.6      |

TODO：case画图展示 @蔡天驰

## OmniLMM 3B
OmniLMM 3B (i.e., MiniCPM-Omni) is an efficient version with promising performance for deployment. The model is built based on SigLip 400M and MiniCPM 2.4B, connected by perceiver resampler. Notable features of OmniLLM 3B include:

- **High Efficiency.** OmniLLM 3B can be efficiently deployed on most GPU cards and personal computers, and even on edge devices such as mobilephones. In terms of visual encoding, we compress the image representations into 64 tokens via perciever resampler, which is significantly fewer than other LMMs based on MLP architecture (typically >512 tokens). This allows OmniLLM 3B to operate with much less memory cost and higher speed during inference.

- **Promising Performance.** OmniLMM 3B achieves state-of-the-art performance on multiple benchmarks (including MMMU, MME and MMbech, etc) among models with comparable sizes, surpassing existing LMMs built on Phi-2. It even achieves comparable or better performance than the 9.6B Qwen-VL-Chat.

- **Bilingual Support.** OmniLMM 3B is the first edge-deployable LMM supporting bilingual mulitmodal interaction in English and Chinese. This is achieved by generalizating mulitmodal capabilites across languages, a technique from our ICLR 2024 spotlight paper.


| **Method**       | #Params | **MME(P)** | **MMB-dev(en)** | **MMB-dev(zh)** | **MMMU-val** | **CMMMU-val** |
|:------------:|:-------:|:----------:|:---------------:|:---------------:|:------------:|:-------------:|
| LLaVA-Phi    | 3B      | 1335       | 59.8            | -               | -            | -             |
| MobileVLM    | 3B      | 1289       | 59.6            | -              | -            | -             |
| Imp-v1       | 3B      | 1434       | 66.5            | -               | -            | -             |
| Qwen-VL-Chat | 9.6B    | **1487**       | 60.6            | 56.7            | **35.9**         | 30.7          |
| OmniLMM 3B | 3B      | 1452       | **67.3**            | **61.9**            | 34.7         | **32.1**          |

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

