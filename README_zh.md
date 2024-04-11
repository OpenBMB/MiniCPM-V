<div align="center">

<!-- <!-- <h1 style="color: #33A6B8; font-family: Helvetica"> OmniLMM </h1> -->

<img src="./assets/minicpmv-omnilmm.png" width="400em" ></img> 

**性能领先且部署高效的多模态大模型**
<p align="center">
  MiniCPM-V 2.0  <a href="https://huggingface.co/openbmb/MiniCPM-V-2.0/">🤗</a> <a href="http://120.92.209.146:80/">🤖</a> |
  OmniLMM-12B <a href="https://huggingface.co/openbmb/OmniLMM-12B/">🤗</a> <a href="http://120.92.209.146:8081">🤖</a>
</p>

</div>


**MiniCPM-V**和**OmniLMM** 是面向图文理解的开源多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出。我们发布了两个版本的模型，旨在实现**领先的性能和高效的部署**：

- **MiniCPM-V 2.8B**：可在终端设备上部署的先进多模态大模型。最新发布的 MiniCPM-V 2.0 可以接受 180 万像素的任意长宽比图像输入，实现了和 Gemini Pro 相近的场景文字识别能力以及和 GPT-4V 相匹的低幻觉率。

- **OmniLMM-12B**：相比同规模其他模型在多个基准测试中具有领先性能，实现了相比 GPT-4V 更低的幻觉率。


[English Document](./README.md)

## 目录 <!-- omit in toc -->
<!-- TOC -->

- [MiniCPM-V 2.8B](#minicpm-v-28b)
- [OmniLMM-12B](#omnilmm-12b)
- [Demo](#demo)
- [安装](#安装)
- [推理](#推理)
  - [模型库](#模型库)
  - [多轮对话](#多轮对话)
  - [Mac 推理](#mac-推理)
  - [手机端部署](#手机端部署)
- [未来计划](#未来计划)

<!-- /TOC -->
<!-- /TOC -->

## MiniCPM-V 2.8B

**MiniCPM-V 2.8B**可以高效部署到终端设备。该模型基于 SigLip-400M 和 [MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/)构建，通过perceiver resampler连接。最新发布的 MiniCPM-V 2.0 的特点包括：

- 🔥 **优秀的性能。**

  MiniCPM-V 2.0 在多个测试基准（如 OCRBench, TextVQA, MME, MMB, MathVista 等）中实现了 7B 以下模型的**最佳性能**。**在综合了 11 个主流多模态大模型评测基准的 OpenCompass 榜单上超过了 Qwen-VL-Chat 9.6B、CogVLM-Chat 17.4B 和 Yi-VL 34B 等更大参数规模的模型**。MiniCPM-V 2.0 还展现出**领先的 OCR 能力**，在场景文字识别能力上**接近 Gemini Pro**，OCRBench 得分达到**开源模型第一**。
  

- 🏆 **可信行为。** 

  多模态大模型深受幻觉问题困扰，模型经常生成和图像中的事实不符的文本。MiniCPM-V 2.0 是 **第一个通过多模态 RLHF 对齐的端侧多模态大模型**（借助 [RLHF-V](https://rlhf-v.github.io/) [CVPR'24] 系列技术）。该模型在 [Object HalBench](https://arxiv.org/abs/2312.00849) 达到**和 GPT-4V 相仿**的性能。


- 🌟 **高清图像高效编码。**

  MiniCPM-V 2.0 可以接受 **180 万像素的任意长宽比图像输入**（基于最新的[LLaVA-UHD](https://arxiv.org/pdf/2403.11703.pdf) 技术），这使得模型可以感知到小物体、密集文字等更加细粒度的视觉信息。 


- ⚡️ **高效部署。**

  MiniCPM-V 2.0 可以**高效部署在大多数 GPU 和个人电脑上**，包括**移动手机等终端设备**。在视觉编码方面，我们通过perceiver resampler将图像表示压缩为更少的 token。这使得 MiniCPM-V 2.0 即便是**面对高分辨率图像，也能占用较低的存储并展现优秀的推理速度**。

- 🙌 **双语支持。**

  MiniCPM-V 2.0 **提供领先的中英双语多模态能力支持**。
  该能力通过 [VisCPM](https://arxiv.org/abs/2308.12038) [ICLR'24] 论文中提出的多模态能力的跨语言泛化技术实现。

### 性能评估 <!-- omit in toc -->

<div align="center">
    <img src=assets/minicpmv-2-peformance.png width=66% />
</div>
<details>
<summary>TextVQA, DocVQA, OCRBench, OpenCompass, MME, MMBench, MMMU, MathVista, LLaVA Bench, Object HalBench 上的详细评测结果。 </summary>
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
* 我们自己评测了正式开源的模型权重。

</details>

### 典型示例 <!-- omit in toc -->


<table align="center">
    <p align="center">
      <img src="assets/minicpmv2-cases_2.png" width=95%/>
    </p>
</table>

我们将 MiniCPM-V 2.0 部署在小米 14 Pro 上，并录制了以下演示视频，未经任何视频剪辑。

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/station.gif" width=36%/>
      <img src="assets/gif_cases/london_car.gif" width=36%/>
    </p>
</table>

### MiniCPM-V 1.0 <!-- omit in toc -->

请参考[这里](./minicpm_v1.md)了解 MiniCPM-V 1.0 的信息和使用教程。


## OmniLMM-12B
**OmniLMM-12B** 是当前系列中性能最佳的版本。该模型基于EVA02-5B和Zephyr-7B-β初始化构建，并使用perceiver resampler连接，采用了课程学习的方法在多模态数据上进行训练。该模型具有三个特点：

- 🔥 **性能领先。**

  OmniLMM-12B 相比其他同规模模型在多个基准测试中取得**领先的性能**（包括 MME、MMBench、SEED-Bench 等），模型掌握了较为丰富的多模态世界知识。

- 🏆 **行为可信。**

  多模态大模型的幻觉问题备受关注，模型经常生成和图像中的事实不符的文本（例如，确信地描述图片中并不存在的物体）。OmniLMM-12B是 **第一个通过多模态 RLHF 对齐的综合能力优秀的开源多模态大模型**（借助 [RLHF-V](https://rlhf-v.github.io/) [CVPR'24] 系列技术）。该模型在 [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench) 幻觉评测基准上达到**开源模型最佳水平**，并在 [Object HalBench](https://arxiv.org/abs/2312.00849) 中**优于GPT-4V**。

- 🕹 **实时多模态交互。**

  我们尝试结合OmniLMM-12B和GPT-3.5 (纯文本模型) ，实现**实时多模态交互助手**。该模型接受来自摄像头的视频流，并借助工具处理语音输入输出。虽然还很初步，我们发现该模型无需视频编辑可以**复现Gemini演示视频中的一些有趣例子**。

### 评测结果 <!-- omit in toc -->

<div align="center">
    <img src=assets/radar_omnilmm12b.png width=66% />
</div>
<details>
<summary> MME, MMBench, MMMU, MMBench, MMHal-Bench, Object HalBench, SeedBench, LLaVA Bench W, MathVista 上的详细评测结果。 </summary>

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
<small>†: 闭源模型</small>
<br>
</details>

### 典型示例 <!-- omit in toc -->

<table align="center" >
  <p align="center" > 
    <img src="assets/omnilmm-12b-examples_2.png" />
  </p>
</table>


我们结合 OmniLMM-12B 和 ChatGPT-3.5 (纯文本模型) 尝试构建 **实时多模态交互助手**. OmniLMM-12B 将视频帧转为对应的图像描述并输入给ChatGPT-3.5来生成对用户指令的响应。演示视频未经编辑。

<div align="center" >
  <video controls src="https://github.com/OpenBMB/OmniLMM/assets/157115220/8fec13bf-bb47-4bf8-8f8c-d0b716a964ec" type="video/mp4" width=80%/>
</div>

## Demo

欢迎通过以下链接使用我们的网页端推理服务： [OmniLMM-12B](http://120.92.209.146:8081) ｜ [MiniCPM-V 2.0](http://120.92.209.146:80).

## 安装

1. 克隆我们的仓库并跳转到相应目录

```bash
git clone https://github.com/OpenBMB/OmniLMM.git
cd OmniLMM
```

1. 创建 conda 环境

```Shell
conda create -n OmniLMM python=3.10 -y
conda activate OmniLMM
```

3. 安装依赖

```shell
pip install -r requirements.txt
```

## 推理

### 模型库

| 模型                | 简介       | 下载链接 |
|:----------------------|:-------------------|:---------------:|
| MiniCPM-V 2.0  | 最新版本，提供高效而领先的端侧双语多模态理解能力。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2.0) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2.0/files) |
| MiniCPM-V  | 第一版 MiniCPM-V    |   [🤗](https://huggingface.co/openbmb/MiniCPM-V) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V/files) |
| OmniLMM-12B | 性能最强的版本                   |  [🤗](https://huggingface.co/openbmb/OmniLMM-12B) &nbsp;&nbsp;  [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/OmniLMM-12B/files) |


### 多轮对话

请参考以下代码使用 `MiniCPM-V` 和 `OmniLMM` 进行推理。

<div align="center">
<img src="assets/hk_OCR.jpg" width="500px">
</div>


```python
from chat import OmniLMMChat, img2base64

chat_model = OmniLMMChat('openbmb/OmniLMM-12B') # or 'openbmb/MiniCPM-V-2'

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

可以得到以下输出:

```
"You should go to the Canon store for a camera."

"The Canon store is located on the right side of the image."
```



### Mac 推理
<details>
<summary>点击查看 MiniCPM-V 2.0 基于Mac MPS运行 (Apple silicon or AMD GPUs)的示例。 </summary>

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2.0', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2.0', trust_remote_code=True)
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
运行:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py
```
</details>


### 手机端部署
MiniCPM-V 2.0 目前可以部署在Android和Harmony操作系统的手机上。 🚀 点击[这里](https://github.com/OpenBMB/mlc-MiniCPM)开始手机端部署。


## 未来计划

- [ ] 支持模型微调
- [ ] 本地用户图形界面部署
- [ ] 实时多模态交互代码开源



## 模型协议 <!-- omit in toc -->

本仓库中代码依照 Apache-2.0 协议开源

OmniLMM 模型权重的使用遵循 “[通用模型许可协议-来源说明-宣传限制-商业授权](https://github.com/OpenBMB/General-Model-License/blob/main/通用模型许可协议-来源说明-宣传限制-商业授权.md)”。

OmniLMM 模型权重对学术研究完全开放。

如需将模型用于商业用途，请联系 cpm@modelbest.cn 来获取书面授权，登记后可以免费商业使用。


## 声明 <!-- omit in toc -->

作为多模态大模型，MiniCPM-V 和 OmniLMM 通过学习大量的多模态数据来生成内容，但它无法理解、表达个人观点或价值判断，它所输出的任何内容都不代表模型开发者的观点和立场。

因此用户在使用 MiniCPM-V 和 OmniLMM 生成的内容时，应自行负责对其进行评估和验证。如果由于使用 OmniLMM 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。


## 机构 <!-- omit in toc -->

本项目由以下机构共同开发：

- <img src="assets/thunlp.png" width="28px"> [清华大学自然语言处理实验室](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [面壁智能](https://modelbest.cn/)
- <img src="assets/zhihu.webp" width="28px"> [知乎](https://www.zhihu.com/ )

