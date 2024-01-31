<div align="center">

# OmniLMM
**Large multi-modal models for strong performance and efficient deployment**
<!-- <p align="center">
  <a href="#-viscpm-chat">multi-modal Conversation Model VisCPM-Chat</a> •
  <a href="#-viscpm-paint">Text-to-image Model VisCPM-Paint</a> •
  <a href="#-inference">Inference</a> •
  <a href="https://arxiv.org/pdf/2308.12038.pdf">Paper</a>
</p> -->

</div>


OmniLMM is a family of open-source large multi-modal models (LMMs) adept at vision & language modeling. The model accepts images and text inputs, and emits text outputs. We release two versions of OmniLMM that are targeted at strong performance and efficient deployment.
- OmniLMM 12B: The most capable version that achieves leading performance among models with comparable sizes on multiple benchmarks.
- OmniLMM 3B: The efficient version that can be deployed on edge devices with promising performance.

## OmniLMM 12B
OmniLMM 12B is the most capable version with strong performance. The model is built based on EVA-E 5B and Zephyr 7B, connected with a perceiver resampler layer, and trained on multi-modal data in a curriculum learning fashion. The model has three notable features:

- **Strong Performance.** OmniLMM 12B achieves leading performance among models with comparable sizes, surpassing established LMMs on multiple benchmarks (including MMMU, MME, MMBench and SEED-Bench, etc). The model also supports OCR capability and endows rich multi-modal world knowledge.

- **Trustworthy Behavior.** LMMs are known for suffering from hallucination, often generating text that is not factually grounded in images (e.g., faithfully describing non-existing objects in images). OmniLMM 12B is the first state-of-the-art open-source LMM aligned via multi-modal RLHF (using our recent [RLHF-V](https://rlhf-v.github.io/) technique) for trustworthy behavior, and ranked #1 among open-source models on MMHalBench and Object Halbench.
  
- **Real-time Multi-modal Interaction.** We combine the OmniLMM 12B and ChatGPT3.5 into a real-time multi-modal interactive assistant. The assistant accepts video stream from the camera and speech stream from microphone, and emits speech output. While still primary, we find the model can replicate some of the fun cases shown in the Gemini Demo video, without any video edition.



<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Size</th>
    <th>MME</th>
    <th>MMMU val</th>
    <th>MMHal-Bench</th>
    <th>SeedBench-I</th>
    <th>LLaVA Bench W</th>
    <th>MathVista</th>
    <th>MMBench dev</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>GPT-4V</td>
    <td>-</td>
    <td>1409</td>
    <td>56.8</td>
    <td>3.53 / 70.8</td>
    <td>71.6 </td>
    <td>93.1 </td>
    <td>47.8 </td>
    <td>75.1 </td>
  </tr>
  <tr>
    <td>Qwen-VL-Chat</td>
    <td>9.6B</td>
    <td>1488</td>
    <td>35.9</td>
    <td>2.93 (59.4)</td>
    <td>64.8 </td>
    <td>67.7 </td>
    <td>33.8 </td>
    <td>60.6 </td>
  </tr>
  <tr>
    <td>CogVLM</td>
    <td>17.4B</td>
    <td>1438</td>
    <td>32.1 </td>
    <td>2.68 (52.1) </td>
    <td>68.8 </td>
    <td>73.9 </td>
    <td>34.7 </td>
    <td>63.7 </td>
  </tr>
  <tr>
    <td>LLaVA 1.5</td>
    <td>13.6B </td>
    <td>1531 </td>
    <td>36.4 </td>
    <td>2.71 (51.0) </td>
    <td>68.1 </td>
    <td>64.6 </td>
    <td>26.4 </td>
    <td>68.2 </td>
  </tr>
  <tr>
    <td>Yi-VL 6B</td>
    <td>6.7B </td>
    <td>- </td>
    <td>39.1 </td>
    <td>- </td>
    <td>66.1 </td>
    <td>39.9 </td>
    <td>28.0 </td>
    <td>68.2 </td>
  </tr>
  <tr>
    <td>OmniLMM-12B</td>
    <td>11.6B </td>
    <td>1637 </td>
    <td>40.7 </td>
    <td>3.45 (68.8) </td>
    <td>71.1 </td>
    <td>72.0 </td>
    <td>34.9 </td>
    <td>71.6 </td>
  </tr>
</tbody>
</table>


<!-- | **Method**       | Size | **MME(P)** | **MMMU val** | MMHal- Bench | SeedBench-I | LLaVA Bench W | MathVista | MMBench dev |
|:------------:|:-------:|:----------:|:---------------:|:---------------:|:------------:|:-------------:|--------------|--------------|
| GPT-4V | - | 1409 | 56.8 | 3.53 (70.8) | 71.6 | 93.1 | 47.8 | 75.1 |
| QWEN-VL-PLUS | - | 1681 | 45.2 | - | 65.7 | 73.7 | 36.0 | 66.2 |
| Qwen-VL-Chat | 9.6B | 1488   | 35.9         | 2.93 (59.4) | 64.8         | 67.7     | 33.8       | 60.6      |
| CogVLM | 17B | 1438 | 32.1 | 2.68 (52.1) | 68.8 | 73.9 | 34.7 | 63.7 |
| LLaVA 1.5 | 14B | 1531 | 36.4 | 2.71 (51.0) | 68.1 | 64.6 | 26.4 | 68.2 |
| Yi-VL | 6.7B | - | 39.1 | - | 66.1 | 39.9 | 28.0 | 68.2 |
| OmniLMM-12B | 12B    | 1637  | 40.7        | 3.45 (68.8) | 71.1        | 72.0      | 34.9      | 71.6      | -->

TODO：case画图展示 @蔡天驰

## OmniLMM 3B
OmniLMM 3B (i.e., MiniCPM-V) is an efficient version with promising performance for deployment. The model is built based on SigLip 400M and [MiniCPM](https://github.com/OpenBMB/MiniCPM)  2.4B, connected by a perceiver resampler layer. Notable features of OmniLLM 3B include:

- **High Efficiency.** OmniLLM 3B can be efficiently deployed on most GPU cards and personal computers, and even on edge devices such as mobile phones. In terms of visual encoding, we compress the image representations into 64 tokens via perceiver resampler, which is significantly fewer than other LMMs based on MLP architecture (typically >512 tokens). This allows OmniLLM 3B to operate with much less memory cost and higher speed during inference.

- **Promising Performance.** OmniLMM 3B achieves state-of-the-art performance on multiple benchmarks (including MMMU, MME and MMbech, etc) among models with comparable sizes, surpassing existing LMMs built on Phi-2. It even achieves comparable or better performance than the 9.6B Qwen-VL-Chat.

- **Bilingual Support.** OmniLMM 3B is the first edge-deployable LMM supporting bilingual multi-modal interaction in English and Chinese. This is achieved by generalizing multi-modal capabilities across languages, a technique from our ICLR 2024 spotlight [paper](https://arxiv.org/abs/2308.12038).


| **Method**       | #Params | **MME(P)** | **MMB-dev(en)** | **MMB-dev(zh)** | **MMMU-val** | **CMMMU-val** |
|:------------:|:-------:|:----------:|:---------------:|:---------------:|:------------:|:-------------:|
| LLaVA-Phi    | 3B      | 1335       | 59.8            | -               | -            | -             |
| MobileVLM    | 3B      | 1289       | 59.6            | -              | -            | -             |
| Imp-v1       | 3B      | 1434       | 66.5            | -               | -            | -             |
| Qwen-VL-Chat | 9.6B    | **1487**       | 60.6            | 56.7            | **35.9**         | 30.7          |
| OmniLMM 3B | 3B      | 1452       | **67.3**            | **61.9**            | 34.7         | **32.1**          |

TODO：视频展示手机端效果？ @蔡天驰

## Demo
Click here to try out the Demo of [OmniLMM 12B](http://120.92.209.146:8081) and [OmniLMM 3B](http://120.92.209.146:80).

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
| OmniLMM-12B | The most capable version with strong performance                                 | [download](https://huggingface.co/openbmb/OmniLMM-12B/blob/main/pytorch_model.v1.bin) |
| OmniLMM-3B  | The efficient version for edge device deployment | [download](https://huggingface.co/openbmb/OmniLMM-3B/blob/main/pytorch_model.v1.bin)  |

### OmniLMM-12B
After downloading the checkpoints, please refer to the following codes to run `OmniLMM` (replace `'/path/to/checkpoint'` with the path of the downloaded checkpoint).

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

TODO：使用文档（安装、使用，包括3B和12B） @朱宏吉

## ✅ TODO

- [ ] Fine-tuning support
- [ ] Local Web-UI deployment
- [ ] Code release for real-time interactive assistant

## 🏫 Institutions

This project is developed by the following institutions:

- <img src="figures/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="figures/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)
- <img src="figures/zhihu.webp" width="28px"> [Zhihu](https://www.zhihu.com/ )

