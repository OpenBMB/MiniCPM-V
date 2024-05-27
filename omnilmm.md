## OmniLMM-12B

> OmniLMM-12B 发布于本项目早期。推荐您使用我们[最新发布的模型](./README_zh.md)，以获得更高效的推理和更强大的性能体验。

> 归档时间：2024-05-19

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

## Online Demo

欢迎通过以下链接使用我们的网页端推理服务： [OmniLMM-12B](http://120.92.209.146:8081) ｜ [MiniCPM-V 2.0](http://120.92.209.146:80).

## 安装

1. 克隆我们的仓库并跳转到相应目录

```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
cd MiniCPM-V
```

1. 创建 conda 环境

```Shell
conda create -n MiniCPMV python=3.10 -y
conda activate MiniCPMV
```

3. 安装依赖

```shell
pip install -r requirements.txt
```

## 推理

### 模型库

| 模型                | 简介       | 下载链接 |
|:----------------------|:-------------------|:---------------:|
| OmniLMM-12B | 性能最强的版本                   |  [🤗](https://huggingface.co/openbmb/OmniLMM-12B) &nbsp;&nbsp;  [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/OmniLMM-12B/files) |

