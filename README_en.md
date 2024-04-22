<div align="center">

<img src="./assets/minicpmv-omnilmm.png" width="400em" ></img> 

**Large multi-modal models for strong performance and efficient deployment**

  <strong>[中文](./README.md) |
  English</strong>

<p align="center">
  MiniCPM-V 2.0  <a href="https://huggingface.co/openbmb/MiniCPM-V-2/">🤗</a> <a href="https://huggingface.co/spaces/openbmb/MiniCPM-V-2">🤖</a> |
  OmniLMM-12B <a href="https://huggingface.co/openbmb/OmniLMM-12B/">🤗</a> <a href="http://120.92.209.146:8081">🤖</a> | <a href="https://openbmb.vercel.app/minicpm-v-2-en"> Technical Blog </a>
</p>

</div>


**MiniCPM-V** and **OmniLMM** are a family of open-source large multimodal models (LMMs) adept at vision & language modeling. The models process images and text inputs and deliver high-quality text outputs. We release two featured versions that are targeted at **strong performance and efficient deployment**:

- **MiniCPM-V 2.8B**: State-of-the-art end-side large multimodal models. Our latest MiniCPM-V 2.0 can accept 1.8 million pixels (e.g., 1344x1344) images at any aspect ratio, and is adept at OCR capability. It achieves comparable performance with Gemini Pro in understanding scene-text and matches GPT-4V in preventing hallucinations.

- **OmniLMM 12B**: The most capable version with leading performance among comparable-sized models on multiple benchmarks. The model also achieves state-of-the-art performance in trustworthy behaviors, with even less hallucination than GPT-4V.



## News <!-- omit in toc -->

* [2024.04.18] We create a HuggingFace Space to host the demo of MiniCPM-V 2.0 at [here](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)!
* [2024.04.17] MiniCPM-V-2.0 supports deploying [WebUI Demo](#webui-demo) now!
* [2024.04.15] MiniCPM-V-2.0 now also supports [fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md) with the SWIFT framework!
* [2024.04.12] We open-source MiniCPM-V-2.0, which achieves comparable performance with Gemini Pro in understanding scene text and outperforms strong Qwen-VL-Chat 9.6B and Yi-VL 34B on <a href="https://rank.opencompass.org.cn/leaderboard-multimodal">OpenCompass</a>, a comprehensive evaluation over 11 popular benchmarks. Click <a href="https://openbmb.vercel.app/minicpm-v-2">here</a> to view the MiniCPM-V 2.0 technical blog.
* [2024.03.14] MiniCPM-V now supports [fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md) with the SWIFT framework. Thanks to [Jintao](https://github.com/Jintao-Huang) for the contribution！
* [2024.03.01] MiniCPM-V now can be deployed on Mac!
* [2024.02.01] We open-source MiniCPM-V and OmniLMM-12B, which support efficient end-side deployment and powerful multimodal capabilities correspondingly.


## Contents <!-- omit in toc -->


- [MiniCPM-V 2.8B](#minicpm-v-28b)
- [OmniLMM-12B](#omnilmm-12b)
- [Online Demo](#online-demo)
- [Install](#install)
- [Inference](#inference)
  - [Model Zoo](#model-zoo)
  - [Multi-turn Conversation](#multi-turn-conversation)
  - [Inference on Mac](#inference-on-mac)
  - [Deployment on Mobile Phone](#deployment-on-mobile-phone)
  - [WebUI Demo](#webui-demo)
- [Finetune](#finetune)
- [TODO](#todo)
- [Citation](#citation)


## MiniCPM-V 2.8B
**MiniCPM-V 2.8B** is an efficient version with promising performance for deployment. The model is built based on SigLip-400M and [MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/), connected by a perceiver resampler. Our latest version, MiniCPM-V 2.0 has several notable features. 

- 🔥 **State-of-the-art Performance.** 

  MiniCPM-V 2.0 achieves **state-of-the-art performance** on multiple benchmarks (including OCRBench, TextVQA, MME, MMB, MathVista, etc) among models under 7B parameters. It even **outperforms strong Qwen-VL-Chat 9.6B, CogVLM-Chat 17.4B, and Yi-VL 34B on OpenCompass, a comprehensive evaluation over 11 popular benchmarks**. Notably, MiniCPM-V 2.0 shows **strong OCR capability**, achieving **comparable performance to Gemini Pro in scene-text understanding**, and **state-of-the-art performance on OCRBench** among open-source models.

- 🏆 **Trustworthy Behavior.** 

  LMMs are known for suffering from hallucination, often generating text not factually grounded in images. MiniCPM-V 2.0 is **the first end-side LMM aligned via multimodal RLHF for trustworthy behavior** (using the recent [RLHF-V](https://rlhf-v.github.io/) [CVPR'24] series technique). This allows the model to **match GPT-4V in preventing hallucinations** on Object HalBench.

- 🌟 **High-Resolution Images at Any Aspect Raito.**

  MiniCPM-V 2.0 can accept **1.8 million pixels (e.g., 1344x1344) images at any aspect ratio**. This enables better perception of fine-grained visual information such as small objects and optical characters, which is achieved via a recent technique from [LLaVA-UHD](https://arxiv.org/pdf/2403.11703.pdf).

- ⚡️ **High Efficiency.** 

  MiniCPM-V 2.0 can be **efficiently deployed on most GPU cards and personal computers**, and **even on end devices such as mobile phones**. For visual encoding, we compress the image representations into much fewer tokens via a perceiver resampler. This allows MiniCPM-V 2.0 to operate with **favorable memory cost and speed during inference even when dealing with high-resolution images**.



- 🙌 **Bilingual Support.** 

  MiniCPM-V 2.0 **supports strong bilingual multimodal capabilities in both English and Chinese**. This is enabled by generalizing multimodal capabilities across languages, a technique from [VisCPM](https://arxiv.org/abs/2308.12038) [ICLR'24].

### Evaluation <!-- omit in toc -->

<div align="center">
    <img src=assets/minicpmv-2-peformance.png width=66% />
</div>
<details>
<summary>Click to view results on TextVQA, DocVQA, OCRBench, OpenCompass, MME, MMBench, MMMU, MathVista, LLaVA Bench, Object HalBench. </summary>
<div align="center">

<table style="margin: 0px auto;">
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th>TextVQA val</th>
    <th>DocVQA test</th>
    <th>OCRBench</th>
    <th>OpenCompass</th>
    <th nowrap="nowrap" >MME</th>
    <th>MMB dev(en)</th>
    <th>MMB dev(zh)</th>
    <th>MMMU val</th>
    <th>MathVista</th>
    <th>LLaVA Bench</th>
    <th nowrap="nowrap">Object HalBench</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td colspan="12" align="left"><strong>Proprietary models</strong></td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">Gemini Pro Vision</td>
    <td>- </td>
    <td>74.6</td>
    <td>88.1</td>
    <td>680</td>
    <td>63.8</td>
    <td>2148.9</td>
    <td>75.2</td>
    <td>74.0</td>
    <td>48.9</td>
    <td>45.8</td>
    <td>79.9</td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">GPT-4V</td>
    <td>- </td>
    <td>78.0</td>
    <td>88.4</td>
    <td>645</td>
    <td>63.2</td>
    <td>1771.5</td>
    <td>75.1</td>
    <td>75.0</td>
    <td>53.8</td>
    <td>47.8</td>
    <td>93.1</td>
    <td>86.4 / 92.7</td>
  </tr>
  <tr>
    <td colspan="12" align="left"><strong>Open-source models 6B~34B</strong></td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >Yi-VL-6B</td>
    <td align="right" >6.7B</td>
    <td>45.5*</td>
    <td>17.1*</td>
    <td>290</td>
    <td>49.3</td>
    <td>1915.1 </td>
    <td>68.6 </td>
    <td>68.3 </td>
    <td>40.3 </td>
    <td>28.8 </td>
    <td>51.9 </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right" >9.6B</td>
    <td>61.5</td>
    <td>62.6</td>
    <td>488 </td>
    <td>52.1 </td>
    <td>1860.0 </td>
    <td>60.6 </td>
    <td>56.7 </td>
    <td>37.0 </td>
    <td>33.8 </td>
    <td>67.7 </td>
    <td>56.2 / 80.0</td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Yi-VL-34B</td>
    <td align="right" >34B</td>
    <td>43.4*</td>
    <td>16.9*</td>
    <td>290</td>
    <td>52.6 </td>
    <td>2050.2</td>
    <td>71.1</td>
    <td>71.4</td>
    <td>45.1</td>
    <td>30.7</td>
    <td>62.3</td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >DeepSeek-VL-7B</td>
    <td align="right" >7.3B</td>
    <td>64.7*</td>
    <td>47.0* </td>
    <td>435</td>
    <td>55.6 </td>
    <td>1765.4 </td>
    <td>74.1 </td>
    <td>72.8 </td>
    <td>38.3 </td>
    <td>36.8</td>
    <td>77.8 </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >TextMonkey</td>
    <td align="right" >9.7B</td>
    <td>64.3</td>
    <td>66.7 </td>
    <td>558</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
    <td>-</td>
    <td>- </td>
    <td>- </td>
  </tr>
    <tr>
    <td  nowrap="nowrap" align="left" >CogVLM-Chat</td>
    <td align="right" >17.4B</td>
    <td>70.4</td>
    <td>33.3*</td>
    <td>590 </td>
    <td>52.5 </td>
    <td>1736.6 </td>
    <td>63.7 </td>
    <td>53.8 </td>
    <td>37.3 </td>
    <td>34.7 </td>
    <td>73.9 </td>
    <td>73.6 / 87.4 </td>
  </tr>
  <tr>
    <td colspan="12" align="left"><strong>Open-source models 1B~3B </strong></td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >DeepSeek-VL-1.3B</td>
    <td align="right" >1.7B</td>
    <td>58.4*</td>
    <td>37.9*</td>
    <td>413</td>
    <td>46.0 </td>
    <td>1531.6 </td>
    <td>64.0 </td>
    <td>61.2 </td>
    <td>33.8 </td>
    <td>29.4 </td>
    <td>51.1 </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >MobileVLM V2</td>
    <td align="right" >3.1B</td>
    <td>57.5</td>
    <td>19.4*</td>
    <td>-</td>
    <td>-</td>
    <td>1440.5(P) </td>
    <td>63.2 </td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >Mini-Gemini</td>
    <td align="right" >2.2B</td>
    <td>56.2</td>
    <td>34.2*</td>
    <td>-</td>
    <td>-</td>
    <td>1653.0 </td>
    <td>59.8 </td>
    <td>- </td>
    <td>31.7 </td>
    <td>-</td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >MiniCPM-V</td>
    <td align="right" >2.8B </td>
    <td>60.6</td>
    <td>38.2 </td>
    <td>366</td>
    <td>47.6</td>
    <td>1650.2 </td>
    <td>67.9 </td>
    <td>65.3 </td>
    <td><strong>38.3</strong></td>
    <td>28.9</td>
    <td>51.3 </td>
    <td>78.4 / 88.5 </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" ><strong>MiniCPM-V 2.0</strong></td>
    <td align="right" >2.8B </td>
    <td><strong>74.1</strong></td>
    <td><strong>71.9</strong> </td>
    <td><strong>605</strong></td>
    <td><strong>55.0</strong></td>
    <td><strong>1808.6</strong> </td>
    <td><strong>69.6</strong> </td>
    <td><strong>68.1</strong> </td>
    <td>38.2 </td>
    <td><strong>38.7</strong></td>
    <td><strong>69.2</strong> </td>
    <td><strong>85.5 / 92.2 </strong></td>
  </tr>
</tbody>
</table>

</div>
* We evaluate the officially released checkpoint by ourselves.

</details>

### Examples <!-- omit in toc -->

<table align="center">
    <p align="center">
      <img src="assets/minicpmv2-cases_2.png" width=95%/>
    </p>
</table>

We deploy MiniCPM-V 2.0 on end devices. The demo video is the raw screen recording on a Xiaomi 14 Pro without edition.

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/station.gif" width=36%/>
      <img src="assets/gif_cases/london_car.gif" width=36%/>
    </p>
</table>

### MiniCPM-V 1.0 <!-- omit in toc -->
Please see the info about MiniCPM-V 1.0 [here](./minicpm_v1.md).


## OmniLMM-12B
**OmniLMM-12B** is the most capable version. The model is built based on EVA02-5B and Zephyr-7B-β, connected with a perceiver resampler layer, and trained on multimodal data in a curriculum fashion. The model has three notable features:

- 🔥 **Strong Performance.** 

  OmniLMM-12B achieves **leading performance** among models with comparable sizes, surpassing established LMMs on multiple benchmarks (including MME, MMBench, SEED-Bench, etc). The model also endows rich multi-modal world knowledge.

- 🏆 **Trustworthy Behavior.** 

  LMMs are known for suffering from hallucination, often generating text that is not factually grounded in images (e.g., faithfully describing non-existing objects in images). OmniLMM-12B is **the first state-of-the-art open-source LMM aligned via multimodal RLHF for trustworthy behavior** (using the recent [RLHF-V](https://rlhf-v.github.io/) technique). It **ranks #1** among open-source models on [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench), and **outperforms GPT-4V** on [Object HalBench](https://arxiv.org/abs/2312.00849).

- 🕹 **Real-time Multimodal Interaction.** 

  We combine the OmniLMM-12B and GPT-3.5 (text-only) into a **real-time multimodal interactive assistant**. The assistant accepts video streams from the camera and speech streams from the microphone and emits speech output. While still primary, we find the model can **replicate some of the fun cases shown in the Gemini Demo video, without any video edition**.


### Evaluation <!-- omit in toc -->
<div align="center">
    <img src=assets/radar_omnilmm12b.png width=66% />
</div>
<details>
<summary>Click to view results on MME, MMBench, MMMU, MMBench, MMHal-Bench, Object HalBench, SeedBench, LLaVA Bench, MathVista. </summary>

<table>
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th>MME</th>
    <th nowrap="nowrap">MMB dev (en)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >MMHal-Bench</th>
    <th nowrap="nowrap" >Object HalBench</th>
    <th nowrap="nowrap" >SeedBench-I</th>
    <th>MathVista</th>
    <th nowrap="nowrap" >LLaVA Bench</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">GPT-4V†</td>
    <td>-</td>
    <td>1771.5</td>
    <td>75.1 </td>
    <td>56.8</td>
    <td>3.53 / 70.8</td>
    <td>86.4 / 92.7</td>
    <td>71.6 </td>
    <td>47.8 </td>
    <td>93.1 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">Qwen-VL-Plus†</td>
    <td>-</td>
    <td>2183.4</td>
    <td>66.2 </td>
    <td>45.2</td>
    <td>- </td>
    <td>- </td>
    <td>65.7 </td>
    <td>36.0 </td>
    <td>73.7 </td>
  </tr>
  <tr>
    <td align="left">Yi-VL 6B</td>
    <td align="right">6.7B </td>
    <td>1915.1 </td>
    <td>68.6 </td>
    <td>40.3 </td>
    <td>- </td>
    <td>- </td>
    <td>67.5 </td>
    <td>28.8 </td>
    <td>51.9 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right">9.6B</td>
    <td>1860.0</td>
    <td>60.6 </td>
    <td>35.9</td>
    <td>2.93 / 59.4</td>
    <td>56.2 / 80.0</td>
    <td>64.8 </td>
    <td>33.8 </td>
    <td>67.7 </td>
  </tr>
  <tr>
    <td align="left" >CogVLM-Chat</td>
    <td align="right">17.4B</td>
    <td>1736.6</td>
    <td>63.7 </td>
    <td>32.1 </td>
    <td>2.68 / 52.1 </td>
    <td>73.6 / 87.4 </td>
    <td>68.8 </td>
    <td>34.7 </td>
    <td>73.9 </td>
  </tr>
  <tr>
    <td align="left" >LLaVA 1.5</td>
    <td align="right">13.6B </td>
    <td>1808.4 </td>
    <td>68.2 </td>
    <td>36.4 </td>
    <td>2.71 / 51.0 </td>
    <td>53.7 / 77.4 </td>
    <td>68.1 </td>
    <td>26.4 </td>
    <td>64.6 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>OmniLMM-12B</b></td>
    <td align="right">11.6B </td>
    <td>1935.8 </td>
    <td>71.6 </td>
    <td>40.7 </td>
    <td>3.45 / 68.8 </td>
    <td>90.3 / 95.5 </td>
    <td>71.1 </td>
    <td>34.9 </td>
    <td>72.0 </td>
  </tr>
</tbody>
</table>
<small>†: Proprietary models</small>
<br>
</details>

### Examples <!-- omit in toc -->

<table align="center" >
  <p align="center" > 
    <img src="assets/omnilmm-12b-examples_2.png" />
  </p>
</table>


We combine the OmniLMM-12B and GPT-3.5 (text-only) into a **real-time multimodal interactive assistant**. Video frames are described in text using OmniLMM-12B, and ChatGPT 3.5 (text-only) is employed to generate response according to the descriptions and user prompts. The demo video is a raw recording without edition. 

<div align="center" >
  <video controls src="https://github.com/OpenBMB/OmniLMM/assets/157115220/485a8f52-fb4d-4eca-8fee-506347efcfc6" type="video/mp4" width=80%/>
</div>


## Online Demo
Click here to try out the Demo of [MiniCPM-V 2.0](http://120.92.209.146:80/) and [OmniLMM-12B](http://120.92.209.146:8081).

## Install

1. Clone this repository and navigate to the source folder

```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
cd MiniCPM-V
```

2. Create conda environment

```Shell
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
```

3. Install dependencies

```shell
pip install -r requirements.txt
```

## Inference

### Model Zoo
| Model                | Description       | Download Link |
|:----------------------|:-------------------|:---------------:|
| MiniCPM-V 2.0  | The latest version for state-of-the-art end-side capabilities with high efficiency.    |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2/files) |
| MiniCPM-V  | The first version of MiniCPM-V.    |  [🤗](https://huggingface.co/openbmb/MiniCPM-V) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V/files) |
| OmniLMM-12B | The most capable version with leading performance.   |  [🤗](https://huggingface.co/openbmb/OmniLMM-12B) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/OmniLMM-12B/files) |

### Multi-turn Conversation
Please refer to the following codes to run `MiniCPM-V` and `OmniLMM`.

<div align="center">
<img src="assets/hk_OCR.jpg" width="500px">
</div>


```python
import torch
from chat import OmniLMMChat, img2base64
torch.manual_seed(0)

chat_model = OmniLMMChat('openbmb/MiniCPM-V-2') # or 'openbmb/OmniLMM-12B'

im_64 = img2base64('./assets/hk_OCR.jpg')

# First round chat 
msgs = [{"role": "user", "content": "Where should I go to buy a camera?"}]

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": answer})
msgs.append({"role": "user", "content": "Where is this store in the image?"})

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)
```

We can obtain the following results:

```
"You should go to the Canon store for a camera."

"The Canon store is located on the right side of the image."
```



### Inference on Mac
<details>
<summary>Click to view an example, to run MiniCPM-V 2.0 on 💻 Mac with MPS (Apple silicon or AMD GPUs). </summary>

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model.eval()

image = Image.open('./assets/hk_OCR.jpg').convert('RGB')
question = 'Where is this photo taken?'
msgs = [{'role': 'user', 'content': question}]

answer, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True
)
print(answer)
```
Run with command:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py
```
</details>

### Deployment on Mobile Phone
Currently MiniCPM-V 2.0 can be deployed on mobile phones with Android and Harmony operating systems. 🚀 Try it out [here](https://github.com/OpenBMB/mlc-MiniCPM).

### WebUI Demo

<details>
<summary>Click to see how to deploy WebUI demo on different devices </summary>
  
```shell
pip install -r requirements.txt
```
  
```shell
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090), run:
python web_demo.py --device cuda --dtype bf16

# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080), run:
python web_demo.py --device cuda --dtype fp16

# For Mac with MPS (Apple silicon or AMD GPUs), run:
PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo.py --device mps --dtype fp16
```
</details>


## Finetune

### MiniCPM-V <!-- omit in toc -->

We now support finetune MiniCPM-V series with the SWIFT framework. SWIFT supports training, inference, evaluation and deployment of nearly 200 LLMs and MLLMs (multimodal large models). It supports the lightweight training solutions provided by PEFT and a complete Adapters Library including techniques such as NEFTune, LoRA+ and LLaMA-PRO.

Best Practices：[MiniCPM-V](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md), [MiniCPM-V-2](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md)



## TODO

- [x] MiniCPM-V fine-tuning support
- [ ] OmniLMM fine-tuning support
- [ ] Code release for real-time interactive assistant

## Model License <!-- omit in toc -->

The code in this repo is released according to [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)

The usage of MiniCPM-V's and OmniLMM's parameters is subject to "[General Model License Agreement - Source Notes - Publicity Restrictions - Commercial License](https://github.com/OpenBMB/General-Model-License/blob/main/通用模型许可协议-来源说明-宣传限制-商业授权.md)"

The parameters are fully open to academic research

Please contact cpm@modelbest.cn to obtain written authorization for commercial uses. Free commercial use is also allowed after registration.

## Statement <!-- omit in toc -->

As LMMs, OmniLMMs generate contents by learning a large amount of multimodal corpora, but they cannot comprehend, express personal opinions or make value judgement. Anything generated by OmniLMMs does not represent the views and positions of the model developers

We will not be liable for any problems arising from the use of OmniLMM open source models, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination or misuse of the model.


## Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)
- <img src="assets/zhihu.webp" width="28px"> [Zhihu](https://www.zhihu.com/ )

## Other Multimodal Projects from Our Team <!-- omit in toc -->

👏 Welcome to explore other multimodal projects of our team:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD)

## Citation

If you find your model/code/paper helpful, please consider cite the following papers:

```bib
@article{yu2023rlhf,
  title={Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback},
  author={Yu, Tianyu and Yao, Yuan and Zhang, Haoye and He, Taiwen and Han, Yifeng and Cui, Ganqu and Hu, Jinyi and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong and others},
  journal={arXiv preprint arXiv:2312.00849},
  year={2023}
}
@article{viscpm,
    title={Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages}, 
    author={Jinyi Hu and Yuan Yao and Chongyi Wang and Shan Wang and Yinxu Pan and Qianyu Chen and Tianyu Yu and Hanghao Wu and Yue Zhao and Haoye Zhang and Xu Han and Yankai Lin and Jiao Xue and Dahai Li and Zhiyuan Liu and Maosong Sun},
    journal={arXiv preprint arXiv:2308.12038},
    year={2023}
}
@article{xu2024llava-uhd,
  title={{LLaVA-UHD}: an LMM Perceiving Any Aspect Ratio and High-Resolution Images},
  author={Xu, Ruyi and Yao, Yuan and Guo, Zonghao and Cui, Junbo and Ni, Zanlin and Ge, Chunjiang and Chua, Tat-Seng and Liu, Zhiyuan and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11703},
  year={2024}
}
```
