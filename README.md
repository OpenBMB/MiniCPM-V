<div align="center">

<img src="./assets/minicpmv.png" width="300em" ></img> 

**A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone**

  <strong>[中文](./README_zh.md) |
  English</strong>

Join our <a href="docs/wechat.md" target="_blank"> 💬 WeChat</a> 


<p align="center">
  MiniCPM-V 2.6 <a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">🤗</a> <a href="https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6">🤖</a> | MiniCPM-Llama3-V 2.5  <a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/">🤗</a> <a href="https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5">🤖</a> |
  <a href=https://arxiv.org/abs/2408.01800>MiniCPM-Llama3-V 2.5 Technical Report</a> 
</p>

</div>


**MiniCPM-V** is a series of end-side multimodal LLMs (MLLMs) designed for vision-language understanding. The models take image, video and text as inputs and provide high-quality text outputs. Since February 2024, we have released 5 versions of the model, aiming to achieve **strong performance and efficient deployment**. The most notable models in this series currently include:

- **MiniCPM-V 2.6**: 🔥🔥🔥 The latest and most capable model in the MiniCPM-V series. With a total of 8B parameters, the model **surpasses GPT-4V in single image, multi-image and video understanding**. It outperforms **GPT-4o mini, Gemini 1.5 Pro and Claude 3.5 Sonnet** in single image understanding, and advances MiniCPM-Llama3-V 2.5's features such as strong OCR capability, trustworthy behavior, multilingual support, and end-side deployment. Due to its superior token density, MiniCPM-V 2.6 can for the first time support real-time video understanding on end-side devices such as iPad.

- **MiniCPM-V 2.0**: The lightest model in the MiniCPM-V series. With 2B parameters, it surpasses larger models such as Yi-VL 34B, CogVLM-Chat 17B, and Qwen-VL-Chat 10B in overall performance. It can accept image inputs of any aspect ratio and up to 1.8 million pixels (e.g., 1344x1344), achieving comparable performance with Gemini Pro in understanding scene-text and matches GPT-4V in low hallucination rates.


## News <!-- omit in toc -->

#### 📌 Pinned
* [2024.08.06] 🔥🔥🔥 We open-source MiniCPM-V 2.6, which outperforms GPT-4V on single image, multi-image and video understanding. It advances popular features of MiniCPM-Llama3-V 2.5, and can support real-time video understanding on iPad. Try it now!
* [2024.08.03] MiniCPM-Llama3-V 2.5 technical report is released! See [here](https://arxiv.org/abs/2408.01800).
* [2024.07.19] MiniCPM-Llama3-V 2.5 supports vLLM now! See [here](#inference-with-vllm).
* [2024.05.28] 🚀🚀🚀 MiniCPM-Llama3-V 2.5 now fully supports its feature in llama.cpp and ollama! Please pull the latest code **of our provided forks** ([llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md), [ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5)). GGUF models in various sizes are available [here](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/tree/main). MiniCPM-Llama3-V 2.5 series is **not supported by the official repositories yet**, and we are working hard to merge PRs. Please stay tuned!
* [2024.05.28] 💫 We now support LoRA fine-tuning for MiniCPM-Llama3-V 2.5, using only 2 V100 GPUs! See more statistics [here](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune#model-fine-tuning-memory-usage-statistics).
* [2024.05.23] 🔍 We've released a comprehensive comparison between Phi-3-vision-128k-instruct and MiniCPM-Llama3-V 2.5, including benchmarks evaluations, multilingual capabilities, and inference efficiency 🌟📊🌍🚀. Click [here](./docs/compare_with_phi-3_vision.md) to view more details.
* [2024.05.23] 🔥🔥🔥 MiniCPM-V tops GitHub Trending and Hugging Face Trending! Our demo, recommended by Hugging Face Gradio’s official account, is available [here](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5). Come and try it out!

<br>

<details> 
<summary>Click to view more news.</summary>

* [2024.06.03] Now, you can run MiniCPM-Llama3-V 2.5 on multiple low VRAM GPUs(12 GB or 16 GB) by distributing the model's layers across multiple GPUs. For more details, Check this [link](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md).
* [2024.05.25] MiniCPM-Llama3-V 2.5 now supports streaming outputs and customized system prompts. Try it [here](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5#usage)!
* [2024.05.24] We release the MiniCPM-Llama3-V 2.5 [gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf), which supports [llama.cpp](#inference-with-llamacpp) inference and provides a 6~8 token/s smooth decoding on mobile phones. Try it now!
* [2024.05.20] We open-soure MiniCPM-Llama3-V 2.5, it has improved OCR capability and supports 30+ languages, representing the first end-side MLLM achieving GPT-4V level performance! We provide [efficient inference](#deployment-on-mobile-phone) and [simple fine-tuning](./finetune/readme.md). Try it now!
* [2024.04.23] MiniCPM-V-2.0 supports vLLM now! Click [here](#inference-with-vllm) to view more details.
* [2024.04.18] We create a HuggingFace Space to host the demo of MiniCPM-V 2.0 at [here](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)!
* [2024.04.17] MiniCPM-V-2.0 supports deploying [WebUI Demo](#webui-demo) now!
* [2024.04.15] MiniCPM-V-2.0 now also supports [fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md) with the SWIFT framework!
* [2024.04.12] We open-source MiniCPM-V 2.0, which achieves comparable performance with Gemini Pro in understanding scene text and outperforms strong Qwen-VL-Chat 9.6B and Yi-VL 34B on <a href="https://rank.opencompass.org.cn/leaderboard-multimodal">OpenCompass</a>, a comprehensive evaluation over 11 popular benchmarks. Click <a href="https://openbmb.vercel.app/minicpm-v-2">here</a> to view the MiniCPM-V 2.0 technical blog.
* [2024.03.14] MiniCPM-V now supports [fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md) with the SWIFT framework. Thanks to [Jintao](https://github.com/Jintao-Huang) for the contribution！
* [2024.03.01] MiniCPM-V now can be deployed on Mac!
* [2024.02.01] We open-source MiniCPM-V and OmniLMM-12B, which support efficient end-side deployment and powerful multimodal capabilities correspondingly.
</details> 


## Contents <!-- omit in toc -->


- [MiniCPM-V 2.6](#minicpm-v-26)
- [MiniCPM-Llama3-V 2.5](#minicpm-llama3-v-25)
- [MiniCPM-V 2.0](#minicpm-v-20)
- [Chat with Our Demo on Gradio 🤗](#chat-with-our-demo-on-gradio-)
- [Install](#install)
- [Inference](#inference)
  - [Model Zoo](#model-zoo)
  - [Multi-turn Conversation](#multi-turn-conversation)
    - [Chat with multiple images](#chat-with-multiple-images)
    - [In-context few-shot learning](#in-context-few-shot-learning)
    - [Chat with video](#chat-with-video)
  - [Inference on Multiple GPUs](#inference-on-multiple-gpus)
  - [Inference on Mac](#inference-on-mac)
  - [Deployment on Mobile Phone](#deployment-on-mobile-phone)
  - [Inference with llama.cpp](#inference-with-llamacpp)
  - [Inference with ollama](#inference-with-ollama)
  - [Inference with vLLM](#inference-with-vllm)
- [Fine-tuning](#fine-tuning)
- [FAQs](#faqs)


## MiniCPM-V 2.6

**MiniCPM-V 2.6** is the latest and most capable model in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-Llama3-V 2.5, and introduces new features for multi-image and video understanding. Notable features of MiniCPM-V 2.6 include:

- 🔥 **Leading Performance.**
  MiniCPM-V 2.6 achieves an average score of 65.2 on the latest version of OpenCompass, a comprehensive evaluation over 8 popular benchmarks. **With only 8B parameters, it surpasses widely used proprietary models like GPT-4o mini, GPT-4V, Gemini 1.5 Pro, and Claude 3.5 Sonnet** for single image understanding.

- 🖼️ **Multi Image Understanding and In-context Learning.** MiniCPM-V 2.6 can also perform **conversation and reasoning over multiple images**. It achieves **state-of-the-art performance** on popular multi-image benchmarks such as Mantis-Eval, BLINK, Mathverse mv and Sciverse mv, and also shows promising in-context learning capability.

- 🎬 **Video Understanding.** MiniCPM-V 2.6 can also **accept video inputs**, performing conversation and providing dense captions for spatial-temporal information. It outperforms **GPT-4V, Claude 3.5 Sonnet and LLaVA-NeXT-Video-34B** on Video-MME with/without subtitles.

- 💪 **Strong OCR Capability and Others.**
  MiniCPM-V 2.6 can process images with any aspect ratio and up to 1.8 million pixels (e.g., 1344x1344). It achieves **state-of-the-art performance on OCRBench, surpassing proprietary models such as GPT-4o, GPT-4V, and Gemini 1.5 Pro**.
  Based on the the latest [RLAIF-V](https://github.com/RLHF-V/RLAIF-V/) and [VisCPM](https://github.com/OpenBMB/VisCPM) techniques, it features **trustworthy behaviors**, with significantly lower hallucination rates than GPT-4o and GPT-4V on Object HalBench, and supports **multilingual capabilities** on English, Chinese, German, French, Italian, Korean, etc.


- 🚀 **Superior Efficiency.**
  In addition to its friendly size, MiniCPM-V 2.6 also shows **state-of-the-art token density** (i.e., number of pixels encoded into each visual token). **It produces only 640 tokens when processing a 1.8M pixel image, which is 75% fewer than most models**. This directly improves the inference speed, first-token latency, memory usage, and power consumption. As a result, MiniCPM-V 2.6 can efficiently support **real-time video understanding** on end-side devices such as iPad.

-  💫  **Easy Usage.**
MiniCPM-V 2.6 can be easily used in various ways: (1) [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpmv-main/examples/llava/README-minicpmv2.6.md) and [ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md) support for efficient CPU inference on local devices, (2) [int4](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4) and [GGUF](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) format quantized models in 16 sizes, (3) [vLLM](#inference-with-vllm) support for high-throughput and memory-efficient inference, (4) fine-tuning on new domains and tasks, (5) quick local WebUI demo setup with [Gradio](#chat-with-our-demo-on-gradio), and (6) online web [demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6).

### Evaluation  <!-- omit in toc -->
<div align="center">
    <img src=assets/radar_final.png width=66% />
</div>

<details>
<summary>Click to view single image results on OpenCompass, MME, MMVet, OCRBench, MMMU, MathVista, MMB, AI2D, TextVQA, DocVQA, HallusionBench, Object HalBench. </summary>
<div align="center">

<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>Token Density<sup>+</sup></th>
            <th>OpenCompass</th>
            <th>MME</th>
            <th>MMVet</th>
            <th>OCRBench</th>
            <th>MMMU val</th>
            <th>MathVista mini</th>
            <th>MMB1.1 test</th>
            <th>AI2D</th>
            <th>TextVQA val</th>
            <th>DocVQA test</th>
            <th>HallusionBench</th>
            <th>Object HalBench</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="15" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o</td>
            <td>-</td>
            <td>1088</td>
            <td>69.9</td>
            <td>2328.7</td>
            <td>69.1</td>
            <td>736</td>
            <td>69.2</td>
            <td>61.3</td>
            <td>82.2</td>
            <td>84.6</td>
            <td>-</td>
            <td>92.8</td>
            <td>55.0</td>
            <td>17.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Claude 3.5 Sonnet</td>
            <td>-</td>
            <td>750</td>
            <td>67.9</td>
            <td>1920.0</td>
            <td>66.0</td>
            <td>788</td>
            <td>65.9</td>
            <td>61.6</td>
            <td>78.5</td>
            <td>80.2</td>
            <td>-</td>
            <td>95.2</td>
            <td>49.9</td>
            <td>13.8</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini 1.5 Pro</td>
            <td>-</td>
            <td>-</td>
            <td>64.4</td>
            <td>2110.6</td>
            <td>64.0</td>
            <td>754</td>
            <td>60.6</td>
            <td>57.7</td>
            <td>73.9</td>
            <td>79.1</td>
            <td>73.5</td>
            <td>86.5</td>
            <td>45.6</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o mini</td>
            <td>-</td>
            <td>1088</td>
            <td>64.1</td>
            <td>2003.4</td>
            <td>66.9</td>
            <td>785</td>
            <td>60.0</td>
            <td>52.4</td>
            <td>76.0</td>
            <td>77.8</td>
            <td>-</td>
            <td>-</td>
            <td>46.1</td>
            <td>12.4</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V</td>
            <td>-</td>
            <td>1088</td>
            <td>63.5</td>
            <td>2070.2</td>
            <td>67.5</td>
            <td>656</td>
            <td>61.7</td>
            <td>54.7</td>
            <td>79.8</td>
            <td>78.6</td>
            <td>78.0</td>
            <td>87.2</td>
            <td>43.9</td>
            <td>14.2</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Step-1V</td>
            <td>-</td>
            <td>-</td>
            <td>59.5</td>
            <td>2206.4</td>
            <td>63.3</td>
            <td>625</td>
            <td>49.9</td>
            <td>44.8</td>
            <td>78.0</td>
            <td>79.2</td>
            <td>71.6</td>
            <td>-</td>
            <td>48.4</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen-VL-Max</td>
            <td>-</td>
            <td>784</td>
            <td>58.3</td>
            <td>2281.7</td>
            <td>61.8</td>
            <td>684</td>
            <td>52.0</td>
            <td>43.4</td>
            <td>74.6</td>
            <td>75.7</td>
            <td>79.5</td>
            <td>93.1</td>
            <td>41.2</td>
            <td>13.4</td>
        </tr>
        <tr>
            <td colspan="15" align="left"><strong>Open-source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Yi-34B</td>
            <td>34B</td>
            <td>157</td>
            <td>55.0</td>
            <td>2006.5</td>
            <td>50.7</td>
            <td>574</td>
            <td>48.8</td>
            <td>40.4</td>
            <td>77.8</td>
            <td>78.9</td>
            <td>69.3</td>
            <td>-</td>
            <td>34.8</td>
            <td>12.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Mini-Gemini-HD-34B</td>
            <td>34B</td>
            <td>157</td>
            <td>-</td>
            <td>2141.0</td>
            <td>59.3</td>
            <td>518</td>
            <td>48.0</td>
            <td>43.3</td>
            <td>-</td>
            <td>80.5</td>
            <td>74.1</td>
            <td>78.9</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Cambrian-34B</td>
            <td>34B</td>
            <td>1820</td>
            <td>58.3</td>
            <td>2049.9</td>
            <td>53.2</td>
            <td>591</td>
            <td>50.4</td>
            <td>50.3</td>
            <td>77.8</td>
            <td>79.5</td>
            <td>76.7</td>
            <td>75.5</td>
            <td>41.6</td>
            <td>14.7</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GLM-4V-9B</td>
            <td>13B</td>
            <td>784</td>
            <td>59.1</td>
            <td>2018.8</td>
            <td>58.0</td>
            <td>776</td>
            <td>46.9</td>
            <td>51.1</td>
            <td>67.9</td>
            <td>71.2</td>
            <td>-</td>
            <td>-</td>
            <td>45.0</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2-8B</td>
            <td>8B</td>
            <td>706</td>
            <td>64.1</td>
            <td>2215.1</td>
            <td>54.3</td>
            <td>794</td>
            <td><strong>51.2</strong></td>
            <td>58.3</td>
            <td><strong>79.4</strong></td>
            <td><strong>83.6</strong></td>
            <td>77.4</td>
            <td><strong>91.6</strong></td>
            <td>45.0</td>
            <td>21.3</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-Llama-V 2.5</td>
            <td>8B</td>
            <td>1882</td>
            <td>58.8</td>
            <td>2024.6</td>
            <td>52.8</td>
            <td>725</td>
            <td>45.8</td>
            <td>54.3</td>
            <td>72.0</td>
            <td>78.4</td>
            <td>76.6</td>
            <td>84.8</td>
            <td>42.4</td>
            <td>10.3</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>2822</strong></td>
            <td><strong>65.2</strong></td>
            <td><strong>2348.4</strong>*</td>
            <td><strong>60.0</strong></td>
            <td><strong>852</strong>*</td>
            <td>49.8*</td>
            <td><strong>60.6</strong></td>
            <td>78.0</td>
            <td>82.1</td>
            <td><strong>80.1<strong></td>
            <td>90.8</td>
            <td><strong>48.1</strong>*</td>
            <td><strong>8.2</strong></td>
        </tr>
    </tbody>
</table>

</div>
* We evaluate this benchmark using chain-of-thought prompting. Specifically, for MME, we used this technique only for the Cognition set.

<sup>+</sup> Token Density: number of pixels encoded into each visual token at maximum resolution, i.e., # pixels at maximum resolution / # visual tokens.

Note: For proprietary models, we calculate token density based on the image encoding charging strategy defined in the official API documentation, which provides an upper-bound estimation.

</details>


<details>
<summary>Click to view multi-image results on Mantis Eval, BLINK, Mathverse mv, Sciverse mv, MIRB.</summary>
<div align="center">
 
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>Mantis Eval</th>
            <th>BLINK val</th>
            <th>Mathverse mv</th>
            <th>Sciverse mv</th>
            <th>MIRB</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="7" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V</td>
            <td>-</td>
            <td>62.7</td>
            <td>54.6</td>
            <td>60.3</td>
            <td>66.9</td>
            <td>53.1</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Interleave-14B</td>
            <td>14B</td>
            <td>66.4</td>
            <td>52.6</td>
            <td>32.7</td>
            <td>30.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="7" align="left"><strong>Open-source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Emu2-Chat</td>
            <td>37B</td>
            <td>37.8</td>
            <td>36.2</td>
            <td>-</td>
            <td>27.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM</td>
            <td>17B</td>
            <td>45.2</td>
            <td>41.1</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">VPG-C</td>
            <td>7B</td>
            <td>52.4</td>
            <td>43.1</td>
            <td>24.3</td>
            <td>23.1</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">VILA 8B</td>
            <td>8B</td>
            <td>51.2</td>
            <td>39.3</td>
            <td>-</td>
            <td>36.5</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternLM-XComposer-2.5</td>
            <td>8B</td>
            <td>53.1*</td>
            <td>48.9</td>
            <td>32.1*</td>
            <td>-</td>
            <td>42.5</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2-8B</td>
            <td>8B</td>
            <td>59.0*</td>
            <td>50.9</td>
            <td>30.5*</td>
            <td>34.4*</td>
            <td><strong>56.9*</strong></td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>69.1</strong></td>
            <td><strong>53.0</strong></td>
            <td><strong>84.9</strong></td>
            <td><strong>74.9</strong></td>
            <td>53.8</td>
        </tr>
    </tbody>
</table>

</div>
* We evaluate the officially released checkpoint by ourselves.
</details>

<details>
<summary>Click to view video results on Video-MME and Video-ChatGPT.</summary>
<div align="center">
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th colspan="2">Video-MME</th>
            <th colspan="5">Video-ChatGPT</th>
        </tr>
        <tr>
            <th align="left"></th>
            <th></th>
            <th>w/o subs</th>
            <th>w subs</th>
            <th>Correctness</th>
            <th>Detail</th>
            <th>Context</th>
            <th>Temporal</th>
            <th>Consistency</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="9" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Claude 3.5 Sonnet</td>
            <td>-</td>
            <td>60.0</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V</td>
            <td>-</td>
            <td>59.9</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="9" align="left"><strong>Open-source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-7B</td>
            <td>7B</td>
            <td>-</td>
            <td>-</td>
            <td>3.39</td>
            <td>3.29</td>
            <td>3.92</td>
            <td>2.60</td>
            <td>3.12</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-34B</td>
            <td>34B</td>
            <td>-</td>
            <td>-</td>
            <td>3.29</td>
            <td>3.23</td>
            <td>3.83</td>
            <td>2.51</td>
            <td>3.47</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM2-Video</td>
            <td>12B</td>
            <td>-</td>
            <td>-</td>
            <td>3.49</td>
            <td><strong>3.46</strong></td>
            <td>3.23</td>
            <td><strong>2.98</strong></td>
            <td><strong>3.64</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LongVA</td>
            <td>7B</td>
            <td>52.4</td>
            <td>54.3</td>
            <td>3.05</td>
            <td>3.09</td>
            <td>3.77</td>
            <td>2.44</td>
            <td><strong>3.64</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2-8B</td>
            <td>8B</td>
            <td>54.0</td>
            <td>56.9</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternLM-XComposer-2.5</td>
            <td>8B</td>
            <td>55.8</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Video</td>
            <td>32B</td>
            <td>60.2</td>
            <td>63.0</td>
            <td>3.48</td>
            <td>3.37</td>
            <td><strong>3.95</strong></td>
            <td>2.64</td>
            <td>3.28</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>60.9</strong></td>
            <td><strong>63.6</strong></td>
            <td><strong>3.59</strong></td>
            <td>3.28</td>
            <td>3.93</td>
            <td>2.73</td>
            <td>3.62</td>
        </tr>
    </tbody>
</table>
</div>
</details>


<details>
<summary>Click to view few-shot results on TextVQA, VizWiz, VQAv2, OK-VQA.</summary>
<div align="center">
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>Shot</th>
            <th>TextVQA val</th>
            <th>VizWiz test-dev</th>
            <th>VQAv2 test-dev</th>
            <th>OK-VQA val</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">Flamingo</td>
            <td rowspan="3">80B</td>
            <td>0*</td>
            <td>35.0</td>
            <td>31.6</td>
            <td>56.3</td>
            <td>40.6</td>
        </tr>
        <tr>
            <td>4</td>
            <td>36.5</td>
            <td>39.6</td>
            <td>63.1</td>
            <td><strong>57.4</strong></td>
        </tr>
        <tr>
            <td>8</td>
            <td>37.3</td>
            <td>44.8</td>
            <td>65.6</td>
            <td>57.5</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">IDEFICS</td>
            <td rowspan="3">80B</td>
            <td>0*</td>
            <td>30.9</td>
            <td>36.0</td>
            <td>60.0</td>
            <td>45.2</td>
        </tr>
        <tr>
            <td>4</td>
            <td>34.3</td>
            <td>40.4</td>
            <td>63.6</td>
            <td>52.4</td>
        </tr>
        <tr>
            <td>8</td>
            <td>35.7</td>
            <td>46.1</td>
            <td>64.8</td>
            <td>55.1</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">OmniCorpus</td>
            <td rowspan="3">7B</td>
            <td>0*</td>
            <td>43.0</td>
            <td>49.8</td>
            <td>63.2</td>
            <td>45.5</td>
        </tr>
        <tr>
            <td>4</td>
            <td>45.4</td>
            <td>51.3</td>
            <td>64.5</td>
            <td>46.5</td>
        </tr>
        <tr>
            <td>8</td>
            <td>45.6</td>
            <td>52.2</td>
            <td>64.7</td>
            <td>46.6</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">Emu2</td>
            <td rowspan="3">37B</td>
            <td>0</td>
            <td>26.4</td>
            <td>40.4</td>
            <td>33.5</td>
            <td>26.7</td>
        </tr>
        <tr>
            <td>4</td>
            <td>48.2</td>
            <td>54.6</td>
            <td>67.0</td>
            <td>53.2</td>
        </tr>
        <tr>
            <td>8</td>
            <td>49.3</td>
            <td>54.7</td>
            <td>67.8</td>
            <td>54.1</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="2">MM1</td>
            <td rowspan="2">30B</td>
            <td>0</td>
            <td>26.2</td>
            <td>40.4</td>
            <td>48.9</td>
            <td>26.7</td>
        </tr>
        <tr>
            <td>8</td>
            <td>49.3</td>
            <td>54.7</td>
            <td><strong>70.9</strong></td>
            <td>54.1</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td align="left" nowrap="nowrap" rowspan="3">MiniCPM-V 2.6<sup>+</sup></td>
            <td rowspan="3">8B</td>
            <td>0</td>
            <td>43.9</td>
            <td>33.8</td>
            <td>45.4</td>
            <td>23.9</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td>4</td>
            <td>63.6</td>
            <td>60.5</td>
            <td>65.5</td>
            <td>50.1</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td>8</td>
            <td><strong>64.6</strong></td>
            <td><strong>63.4</strong></td>
            <td>68.2</td>
            <td>51.4</td>
        </tr>
    </tbody>
</table>


</div>
* denotes zero image shot and two additional text shots following Flamingo.

<sup>+</sup> We evaluate the pretraining ckpt without SFT.
</details>

### Examples <!-- omit in toc -->

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmv2_6/multi_img-bike.png" alt="Bike" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/multi_img-menu.png" alt="Menu" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/multi_img-code.png" alt="Code" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/ICL-Mem.png" alt="Mem" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/multiling-medal.png" alt="medal" style="margin-bottom: 10px;">
</div>
<details>
  <summary>Click to view more cases.</summary>
  <div style="display: flex; flex-direction: column; align-items: center;">
    <img src="assets/minicpmv2_6/ICL-elec.png" alt="elec" style="margin-bottom: 5px;">
    <img src="assets/minicpmv2_6/multiling-olympic.png" alt="Menu" style="margin-bottom: 10px;">
  </div>
</details>

We deploy MiniCPM-V 2.6 on end devices. The demo video is the raw screen recording on a iPad Pro without edition.

<table align="center"> 
    <p align="center">
      <img src="assets/gif_cases/ai.gif" width=32%/>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="assets/gif_cases/beer.gif" width=32%/>
    </p>
</table> 

<table align="center"> 
    <p align="center">
      <img src="assets/gif_cases/ticket.gif" width=32%/>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="assets/gif_cases/wfh.gif" width=32%/>
    </p>
</table> 

<table align="center">
    <p align="center">
      <video src="https://github.com/user-attachments/assets/21f4b818-ede1-4822-920e-91281725c830" width="360" /> </video>
      <!-- <video src="https://github.com/user-attachments/assets/c835f757-206b-4d9c-8e36-70d67b453628" width="360" /> </video> -->
    </p>
</table>

## MiniCPM-Llama3-V 2.5

<details>
<summary>Click to view more details of MiniCPM-Llama3-V 2.5</summary>

**MiniCPM-Llama3-V 2.5** is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Llama3-8B-Instruct with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-V 2.0. Notable features of MiniCPM-Llama3-V 2.5 include:

- 🔥 **Leading Performance.**
  MiniCPM-Llama3-V 2.5 has achieved an average score of 65.1 on OpenCompass, a comprehensive evaluation over 11 popular benchmarks. **With only 8B parameters, it surpasses widely used proprietary models like GPT-4V-1106, Gemini Pro, Claude 3 and Qwen-VL-Max** and greatly outperforms other Llama 3-based MLLMs.

- 💪 **Strong OCR Capabilities.**
  MiniCPM-Llama3-V 2.5 can process images with any aspect ratio and up to 1.8 million pixels (e.g., 1344x1344), achieving a **700+ score on OCRBench, surpassing proprietary models such as GPT-4o, GPT-4V-0409, Qwen-VL-Max and Gemini Pro**. Based on recent user feedback, MiniCPM-Llama3-V 2.5 has now enhanced full-text OCR extraction, table-to-markdown conversion, and other high-utility capabilities, and has further strengthened its instruction-following and complex reasoning abilities, enhancing multimodal interaction experiences.

- 🏆 **Trustworthy Behavior.**
  Leveraging the latest [RLAIF-V](https://github.com/RLHF-V/RLAIF-V/) method (the newest technique in the [RLHF-V](https://github.com/RLHF-V) [CVPR'24] series), MiniCPM-Llama3-V 2.5 exhibits more trustworthy behavior. It achieves a **10.3%** hallucination rate on Object HalBench, lower than GPT-4V-1106 (13.6%), achieving the best-level performance within the open-source community. [Data released](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset).

- 🌏 **Multilingual Support.**
  Thanks to the strong multilingual capabilities of Llama 3 and the cross-lingual generalization technique from [VisCPM](https://github.com/OpenBMB/VisCPM), MiniCPM-Llama3-V 2.5 extends its bilingual (Chinese-English) multimodal capabilities to **over 30 languages including German, French, Spanish, Italian, Korean etc.** [All Supported Languages](./assets/minicpm-llama-v-2-5_languages.md).

- 🚀 **Efficient Deployment.**
  MiniCPM-Llama3-V 2.5 systematically employs **model quantization, CPU optimizations, NPU optimizations and compilation optimizations**, achieving high-efficiency deployment on end-side devices. For mobile phones with Qualcomm chips, we have integrated the NPU acceleration framework QNN into llama.cpp for the first time. After systematic optimization, MiniCPM-Llama3-V 2.5 has realized a **150x acceleration in end-side MLLM image encoding** and a **3x speedup in language decoding**.

-  💫  **Easy Usage.**
MiniCPM-Llama3-V 2.5 can be easily used in various ways: (1) [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md) and [ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5) support for efficient CPU inference on local devices, (2) [GGUF](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf) format quantized models in 16 sizes, (3) efficient [LoRA](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune#lora-finetuning) fine-tuning with only 2 V100 GPUs, (4) [streaming output](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5#usage), (5) quick local WebUI demo setup with [Gradio](https://github.com/OpenBMB/MiniCPM-V/blob/main/web_demo_2.5.py) and [Streamlit](https://github.com/OpenBMB/MiniCPM-V/blob/main/web_demo_streamlit-2_5.py), and (6) interactive demos on [HuggingFace Spaces](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5).

### Evaluation  <!-- omit in toc -->

<div align="center">
    <img src=assets/MiniCPM-Llama3-V-2.5-peformance.png width=66% />
</div>
<details>
<summary>Click to view results on TextVQA, DocVQA, OCRBench, OpenCompass, MME, MMBench, MMMU, MathVista, LLaVA Bench, RealWorld QA, Object HalBench. </summary>
<div align="center">

<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>OCRBench</th>
            <th>TextVQA val</th>
            <th>DocVQA test</th>
            <th>Open-Compass</th>
            <th>MME</th>
            <th>MMB test (en)</th>
            <th>MMB test (cn)</th>
            <th>MMMU val</th>
            <th>Math-Vista</th>
            <th>LLaVA Bench</th>
            <th>RealWorld QA</th>
            <th>Object HalBench</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="14" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini Pro</td>
            <td>-</td>
            <td>680</td>
            <td>74.6</td>
            <td>88.1</td>
            <td>62.9</td>
            <td>2148.9</td>
            <td>73.6</td>
            <td>74.3</td>
            <td>48.9</td>
            <td>45.8</td>
            <td>79.9</td>
            <td>60.4</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V (2023.11.06)</td>
            <td>-</td>
            <td>645</td>
            <td>78.0</td>
            <td>88.4</td>
            <td>63.5</td>
            <td>1771.5</td>
            <td>77.0</td>
            <td>74.4</td>
            <td>53.8</td>
            <td>47.8</td>
            <td>93.1</td>
            <td>63.0</td>
            <td>86.4</td>
        </tr>
        <tr>
            <td colspan="14" align="left"><strong>Open-source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Mini-Gemini</td>
            <td>2.2B</td>
            <td>-</td>
            <td>56.2</td>
            <td>34.2*</td>
            <td>-</td>
            <td>1653.0</td>
            <td>-</td>
            <td>-</td>
            <td>31.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen-VL-Chat</td>
            <td>9.6B</td>
            <td>488</td>
            <td>61.5</td>
            <td>62.6</td>
            <td>51.6</td>
            <td>1860.0</td>
            <td>61.8</td>
            <td>56.3</td>
            <td>37.0</td>
            <td>33.8</td>
            <td>67.7</td>
            <td>49.3</td>
            <td>56.2</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">DeepSeek-VL-7B</td>
            <td>7.3B</td>
            <td>435</td>
            <td>64.7*</td>
            <td>47.0*</td>
            <td>54.6</td>
            <td>1765.4</td>
            <td>73.8</td>
            <td>71.4</td>
            <td>38.3</td>
            <td>36.8</td>
            <td>77.8</td>
            <td>54.2</td>
            <td>-</td>
        </tr>        
        <tr>
            <td nowrap="nowrap" align="left">Yi-VL-34B</td>
            <td>34B</td>
            <td>290</td>
            <td>43.4*</td>
            <td>16.9*</td>
            <td>52.2</td>
            <td><strong>2050.2</strong></td>
            <td>72.4</td>
            <td>70.7</td>
            <td>45.1</td>
            <td>30.7</td>
            <td>62.3</td>
            <td>54.8</td>
            <td>79.3</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM-Chat</td>
            <td>17.4B</td>
            <td>590</td>
            <td>70.4</td>
            <td>33.3*</td>
            <td>54.2</td>
            <td>1736.6</td>
            <td>65.8</td>
            <td>55.9</td>
            <td>37.3</td>
            <td>34.7</td>
            <td>73.9</td>
            <td>60.3</td>
            <td>73.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">TextMonkey</td>
            <td>9.7B</td>
            <td>558</td>
            <td>64.3</td>
            <td>66.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
          <td nowrap="nowrap" align="left">Idefics2</td>
          <td>8.0B</td>
          <td>-</td>
          <td>73.0</td>
          <td>74.0</td>
          <td>57.2</td>
          <td>1847.6</td>
          <td>75.7</td>
          <td>68.6</td>
          <td>45.2</td>
          <td>52.2</td>
          <td>49.1</td>
          <td>60.7</td>
          <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Bunny-LLama-3-8B</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>54.3</td>
            <td>1920.3</td>
            <td>77.0</td>
            <td>73.9</td>
            <td>41.3</td>
            <td>31.5</td>
            <td>61.2</td>
            <td>58.8</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT Llama-3-8B</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>78.2</td>
            <td>-</td>
            <td>1971.5</td>
            <td>-</td>
            <td>-</td>
            <td>41.7</td>
            <td>37.5</td>
            <td>80.1</td>
            <td>60.0</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Phi-3-vision-128k-instruct</td>
            <td>4.2B</td>
            <td>639*</td>
            <td>70.9</td>
            <td>-</td>
            <td>-</td>
            <td>1537.5*</td>
            <td>-</td>
            <td>-</td>
            <td>40.4</td>
            <td>44.5</td>
            <td>64.2*</td>
            <td>58.8*</td>
            <td>-</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 1.0</td>
            <td>2.8B</td>
            <td>366</td>
            <td>60.6</td>
            <td>38.2</td>
            <td>47.5</td>
            <td>1650.2</td>
            <td>64.1</td>
            <td>62.6</td>
            <td>38.3</td>
            <td>28.9</td>
            <td>51.3</td>
            <td>51.2</td>
            <td>78.4</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.0</td>
            <td>2.8B</td>
            <td>605</td>
            <td>74.1</td>
            <td>71.9</td>
            <td>54.5</td>
            <td>1808.6</td>
            <td>69.1</td>
            <td>66.5</td>
            <td>38.2</td>
            <td>38.7</td>
            <td>69.2</td>
            <td>55.8</td>
            <td>85.5</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-Llama3-V 2.5</td>
            <td>8.5B</td>
            <td><strong>725</strong></td>
            <td><strong>76.6</strong></td>
            <td><strong>84.8</strong></td>
            <td><strong>65.1</strong></td>
            <td>2024.6</td>
            <td><strong>77.2</strong></td>
            <td><strong>74.2</strong></td>
            <td><strong>45.8</strong></td>
            <td><strong>54.3</strong></td>
            <td><strong>86.7</strong></td>
            <td><strong>63.5</strong></td>
            <td><strong>89.7</strong></td>
        </tr>
    </tbody>
</table>


</div>
* We evaluate the officially released checkpoint by ourselves.

</details>

<div align="center">
    <img src="assets/llavabench_compare_3.png" width="100%" />
    <br>
    Evaluation results of multilingual LLaVA Bench
</div>

### Examples <!-- omit in toc -->

<table align="center" >
  <p align="center" > 
  <img src="assets/minicpmv-llama3-v2.5/cases_all.png" />
  </p>
</table>

</details>


## MiniCPM-V 2.0

<details>
<summary>Click to view more details of MiniCPM-V 2.0</summary>


**MiniCPM-V 2.0** is an efficient version with promising performance for deployment. The model is built based on SigLip-400M and [MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/), connected by a perceiver resampler. Our latest version, MiniCPM-V 2.0 has several notable features. 

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

</details>

## Legacy Models <!-- omit in toc --> 

| Model                | Introduction and Guidance       |
|:----------------------|:-------------------:|
| MiniCPM-V 1.0  | [Document](./minicpm_v1.md)   | 
| OmniLMM-12B  | [Document](./omnilmm_en.md)   |  


## Chat with Our Demo on Gradio 🤗

We provide online and local demos powered by Hugging Face Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>, the most popular model deployment framework nowadays. It supports streaming outputs, progress bars, queuing, alerts,  and other useful features.


### Online Demo <!-- omit in toc --> 

Click here to try out the online demo of [MiniCPM-V 2.6](https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2).

### Local WebUI Demo <!-- omit in toc --> 
  
You can easily build your own local WebUI demo with Gradio using the following commands.
  
```shell
pip install -r requirements.txt
```
  
```shell
# For NVIDIA GPUs, run:
python web_demo_2.6.py --device cuda

```


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

| Model           | Device | Memory    | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Description       | Download |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 2.6| GPU | 17 GB  | The latest version, achieving state-of-the-art end-side performance for single image, multi-image and video understanding.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6) |
| MiniCPM-V 2.6 gguf | CPU | 6 GB  | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6-gguf) |
| MiniCPM-V 2.6 int4 | GPU | 7 GB  | The int4 quantized version, lower GPU memory usage.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6-int4) |
| MiniCPM-Llama3-V 2.5 | GPU | 19 GB | Strong end-side multimodal performance.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5) |
| MiniCPM-Llama3-V 2.5 gguf | CPU  | 6 GB | The gguf version, lower memory usage and faster inference.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf) &nbsp;&nbsp;[<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-gguf) |
| MiniCPM-Llama3-V 2.5 int4 | GPU | 8 GB | The int4 quantized version, lower GPU memory usage. |  [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4/) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-int4) |
| MiniCPM-V 2.0 | GPU | 8 GB | Light version, balance the performance the computation cost.   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2) |
| MiniCPM-V 1.0 | GPU | 7 GB | Lightest version, achieving the fastest inference. |   [🤗](https://huggingface.co/openbmb/MiniCPM-V) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V) |

### Multi-turn Conversation

Please refer to the following codes to run.

<div align="center">
<img src="assets/airplane.jpeg" width="500px">
</div>


```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('./assets/airplane.jpeg').convert('RGB')

# First round chat 
question = "Tell me the model of this aircraft."
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```

You will get the following output:

```
"The aircraft in the image is an Airbus A380, which can be identified by its large size, double-deck structure, and the distinctive shape of its wings and engines. The A380 is a wide-body aircraft known for being the world's largest passenger airliner, designed for long-haul flights. It has four engines, which are characteristic of large commercial aircraft. The registration number on the aircraft can also provide specific information about the model if looked up in an aviation database."

"The Airbus A380 is a double-deck, wide-body, four-engine jet airliner made by Airbus. It is the world's largest passenger airliner and is known for its long-haul capabilities. The aircraft was developed to improve efficiency and comfort for passengers traveling over long distances. It has two full-length passenger decks, which can accommodate more passengers than a typical single-aisle airplane. The A380 has been operated by airlines such as Lufthansa, Singapore Airlines, and Emirates, among others. It is widely recognized for its unique design and significant impact on the aviation industry."
```

#### Chat with multiple images
<details>
<summary> Click to view Python code running MiniCPM-V 2.6 with multiple images input. </summary>
  
```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image1 = Image.open('image1.jpg').convert('RGB')
image2 = Image.open('image2.jpg').convert('RGB')
question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'

msgs = [{'role': 'user', 'content': [image1, image2, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```
</details>

#### In-context few-shot learning
<details>
<summary> Click to view Python code running MiniCPM-V 2.6 with few-shot input. </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

question = "production date" 
image1 = Image.open('example1.jpg').convert('RGB')
answer1 = "2023.08.04"
image2 = Image.open('example2.jpg').convert('RGB')
answer2 = "2007.04.24"
image_test = Image.open('test.jpg').convert('RGB')

msgs = [
    {'role': 'user', 'content': [image1, question]}, {'role': 'assistant', 'content': [answer1]},
    {'role': 'user', 'content': [image2, question]}, {'role': 'assistant', 'content': [answer2]},
    {'role': 'user', 'content': [image_test, question]}
]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```
</details>

#### Chat with video
<details>
<summary> Click to view Python code running MiniCPM-V 2.6 with video input. </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path="video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]}, 
]

# Set decode params for video
params = {}
params["use_image_id"] = False
params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)
print(answer)
```
</details>


### Inference on Multiple GPUs
You can run MiniCPM-Llama3-V 2.5 on multiple low VRAM GPUs (12 GB or 16 GB) by distributing the model's layers across multiple GPUs. Please refer to this [tutorial](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md) for detailed instructions on how to load the model and inference using multiple low VRAM GPUs.


### Inference on Mac
<details>
<summary>Click to view an example, to run MiniCPM-Llama3-V 2.5 on 💻 Mac with MPS (Apple silicon or AMD GPUs). </summary>

```python
# test.py  Need more than 16GB memory.
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, low_cpu_mem_usage=True)
model = model.to(device='mps')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
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
MiniCPM-V 2.0 can be deployed on mobile phones with Android operating systems. 🚀 Click [MiniCPM-V 2.0](https://github.com/OpenBMB/mlc-MiniCPM) to install apk.

### Inference with llama.cpp
MiniCPM-V 2.6 can run with llama.cpp now! See [our fork of llama.cpp](https://github.com/OpenBMB/llama.cpp/tree/minicpmv-main/examples/llava/README-minicpmv2.6.md) for more detail. This implementation supports smooth inference of 16~18 token/s on iPad (test environment：iPad Pro + M4).

### Inference with ollama
MiniCPM-V 2.6 can run with ollama now! See [our fork of ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md) for more detail. This implementation supports smooth inference of 16~18 token/s on iPad (test environment：iPad Pro + M4).

### Inference with vLLM

<details>
<summary> vLLM now officially supports MiniCPM-V 2.6, MiniCPM-Llama3-V 2.5 and MiniCPM-V 2.0, Click to see. </summary>

1. Install vLLM(>=0.5.4):
```shell
pip install vllm
```
2. Install timm: (optional, MiniCPM-V 2.0 need timm)
```shell
pip install timm==0.9.10
```
3. Run the example(for image):
```python
from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-2_6"
# Also available for previous models
# MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"
# MODEL_NAME = "HwwwH/MiniCPM-V-2"

image = Image.open("xxx.png").convert("RGB")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=1,
    max_model_len=2048
)

messages = [{
    "role":
    "user",
    "content":
    # Number of images
    "(<image>./</image>)" + \
    "\nWhat is the content of this image?" 
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Single Inference
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
        # Multi images, the number of images should be equal to that of `(<image>./</image>)`
        # "image": [image, image] 
    },
}
# Batch Inference
# inputs = [{
#     "prompt": prompt,
#     "multi_modal_data": {
#         "image": image
#     },
# } for _ in 2]


# 2.6
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
# stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    use_beam_search=True,
    temperature=0, 
    best_of=3,
    max_tokens=1024
)

outputs = llm.generate(inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```
4. click [here](https://modelbest.feishu.cn/wiki/C2BWw4ZP0iCDy7kkCPCcX2BHnOf?from=from_copylink) if you want to use it with *video*, or get more details about `vLLM`.
</details>

## Fine-tuning

### Simple Fine-tuning <!-- omit in toc -->

We support simple fine-tuning with Hugging Face for MiniCPM-V 2.0 and MiniCPM-Llama3-V 2.5.

[Reference Document](./finetune/readme.md)

### With the SWIFT Framework <!-- omit in toc -->

We now support MiniCPM-V series fine-tuning with the SWIFT framework. SWIFT supports training, inference, evaluation and deployment of nearly 200 LLMs and MLLMs . It supports the lightweight training solutions provided by PEFT and a complete Adapters Library including techniques such as NEFTune, LoRA+ and LLaMA-PRO.

Best Practices：[MiniCPM-V 1.0](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md), [MiniCPM-V 2.0](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md)

## FAQs
Click here to view the [FAQs](./docs/faqs.md)

## Model License <!-- omit in toc -->

* This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 

* The usage of MiniCPM-V model weights must strictly follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).

* The models and weights of MiniCPM are completely free for academic research. after filling out a ["questionnaire"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g) for registration, are also available for free commercial use.
  

## Statement <!-- omit in toc -->

As LMMs, MiniCPM-V models (including OmniLMM) generate contents by learning a large amount of multimodal corpora, but they cannot comprehend, express personal opinions or make value judgement. Anything generated by MiniCPM-V models does not represent the views and positions of the model developers

We will not be liable for any problems arising from the use of MiniCPM-V models, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination or misuse of the model.


## Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)
- <img src="assets/zhihu.webp" width="28px"> [Zhihu](https://www.zhihu.com/ )

## 🌟 Star History <!-- omit in toc -->


<table align="center">
    <p align="center">
      <img src="assets/star_history.svg"/>
    </p>
</table>

<!-- <picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=OpenBMB/MiniCPM-V&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=OpenBMB/MiniCPM-V&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=OpenBMB/MiniCPM-V&type=Date"
  />
</picture> -->

## Key Techniques and Other Multimodal Projects <!-- omit in toc -->

👏 Welcome to explore key techniques of MiniCPM-V and other multimodal projects of our team:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)


## Citation <!-- omit in toc -->

If you find our model/code/paper helpful, please consider cite our papers 📝 and star us ⭐️！

```bib
@article{yao2024minicpmv,
      title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone}, 
      author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and Chen, Qianyu and Zhou, Huarong and Zou, Zhensheng and Zhang, Haoye and Hu, Shengding and Zheng, Zhi and Zhou, Jie and Cai, Jie and Han, Xu and Zeng, Guoyang and Li, Dahai and Liu, Zhiyuan and Sun, Maosong},
      journal={arXiv preprint 2408.01800},
      year={2024},
}
```
