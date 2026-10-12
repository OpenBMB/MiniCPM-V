## MiniCPM-V 4.6

> 归档日期：2026-10-10

[English](./minicpm_v4dot6_en.md) | [简体中文](./minicpm_v4dot6_zh.md) | [返回 README](../README_zh.md)

[在线体验](https://huggingface.co/spaces/openbmb/MiniCPM-V-4.6-Demo)


**MiniCPM-V 4.6** 是 MiniCPM-V 系列的最新模型，也是迄今最端侧友好的一代模型。该模型基于 SigLIP2-400M 和 Qwen3.5-0.8B LLM 构建。其延续了 MiniCPM-V 在单图、多图和视频理解方面的强大能力，同时显著提升了计算效率，还首次支持了 4倍/16倍 混合视觉 token 压缩率，其主要特点包括：

- 🔥 **领先的基础能力。**
  MiniCPM-V 4.6 在 Artificial Analysis Intelligence Index 基准上获得 13 分，以 19 倍更低的 token 成本超过 Qwen3.5-0.8B 的 10 分，并以 43 倍更低的 token 成本超过 Qwen3.5-0.8B-Thinking 的 11 分。同时，它也超过了参数规模更大的 Ministral 3 3B（11 分）。
- 💪 **出色的多模态能力。**
  MiniCPM-V 4.6 在绝大多数图文理解任务上优于 Qwen3.5-0.8B，并在 OpenCompass、RefCOCO、HallusionBench、MUIRBench、OCRBench 等众多评测基准上展现出 Qwen3.5 2B 级别的能力。
- 🚀 **极致高效架构。**
  MiniCPM-V 4.6 基于 [LLaVA-UHD v4](https://github.com/THUMAI-Lab/LLaVA-UHD-v4)，引入 ViT 内部视觉 token 早压缩机制，将视觉编码阶段计算量降低 50% 以上，在效率上甚至超越部分更小的模型，相比 Qwen3.5-0.8B 实现约 1.5 倍的 token 吞吐；同时支持 4 倍/16 倍混合视觉 token 压缩率，在精度与速度之间灵活切换。
- 📱 **广泛的手机平台支持。**
  MiniCPM-V 4.6 可在 iOS、安卓、鸿蒙三大主流手机平台完成部署，并开源配备了端侧适配代码，开发者可在自己的设备上[一键复现端侧体验](#ios安卓鸿蒙端侧平台推理)。
- 🛠️ **开发者友好。**
  MiniCPM-V 4.6 适配 SGLang、vLLM、llama.cpp、Ollama 等[推理框架](#训练和推理框架与-api-支持)，并支持 SWIFT、LLaMA-Factory 等[微调生态](#训练和推理框架与-api-支持)。开发者可以在消费级显卡上为新领域、新任务快速定制模型。我们还提供了覆盖 GGUF、BNB、AWQ、GPTQ 格式的多种量化版本权重，适配多样的部署需求。


### 性能评估 <!-- omit in toc -->


**综合性能（Instruct）**

<p align="center">
  <img src="../assets/minicpmv4.6/instruct-zh.png" width="90%"></img>
</p>

<details>
<summary>点击查看 MiniCPM-V 4.6-Thinking 的综合性能。</summary>

<p align="center">
  <img src="../assets/minicpmv4.6/thinking-zh.png" width="90%"></img>
</p>

</details>
<br>

**MiniCPM-V 4.6 推理效率**
<table align="center">
  <tr>
    <td align="center" width="50%"><b>高并发请求吞吐量</b></td>
    <td align="center" width="50%"><b>单并发请求首响延迟 TTFT (ms)</b></td>
  </tr>
  <tr>
    <td align="center" valign="middle"><img src="../assets/minicpmv4.6/throughput-zh.png" width="110%"></img></td>
    <td align="center" valign="middle"><img src="../assets/minicpmv4.6/ttft-zh.png" width="100%"></img></td>
  </tr>
</table>


### 典型示例 <!-- omit in toc -->

<div align="center">
  <a href="https://www.youtube.com/watch?v=Ch5UG1FoysM"><img src="../assets/minicpmv4.6/video_play.png" width="70%"></a>
</div>

MiniCPM-V 4.6 可以在 **iOS、安卓、鸿蒙** 等主流端侧平台完成部署。

<table align="center">
  <tr>
    <td align="center"><b>iPhone</b><br><sub>iPhone 17 Pro Max</sub></td>
    <td align="center"><b>安卓</b><br><sub>红米 K70</sub></td>
    <td align="center"><b>鸿蒙</b><br><sub>华为 nova 14</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="../assets/minicpmv4.6/v46_iphone_cn_handwriting.gif" width="100%"/></td>
    <td align="center"><img src="../assets/minicpmv4.6/v46_android_cn_refraction.gif" width="100%"/></td>
    <td align="center"><img src="../assets/minicpmv4.6/v46_harmonyos_cn_menu.gif" width="100%"/></td>
  </tr>
</table>

### 使用说明

#### 基于 Transformers 推理 <!-- omit in toc -->
<details>
<summary>点击展开基于 Transformers 的推理示例。</summary>

##### 安装 <!-- omit in toc -->

```bash
pip install "transformers[torch]>=5.7.0" torchvision torchcodec
```

> **CUDA 兼容性提示：** `torchcodec`（用于视频解码）可能与部分 CUDA 版本存在兼容性问题。例如 `torch>=2.11` 默认使用 CUDA 13.1，在 CUDA 12.x 环境下可能出现 `RuntimeError: Could not load libtorchcodec` 等错误。两种解决方案：
>
> 1. **用 `PyAV` 替代 `torchcodec`** —— 图像和视频推理均可正常使用，无 CUDA 版本限制：
>    ```bash
>    pip install "transformers[torch]>=5.7.0" torchvision av
>    ```
> 2. **安装 torch 时指定 CUDA 版本**以匹配当前环境（如 CUDA 12.8）：
>    ```bash
>    pip install "transformers>=5.7.0" torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
>    ```


##### 加载模型 <!-- omit in toc -->

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4.6"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# 推荐使用 Flash Attention 2 以获得更好的加速与显存节省，
# 尤其在多图和视频场景下效果显著。
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
```

##### 图片推理 <!-- omit in toc -->

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"},
            {"type": "text", "text": "What causes this phenomenon?"},
        ],
    }
]

downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
    downsample_mode=downsample_mode,
    max_slice_nums=36,
).to(model.device)

generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

##### 视频推理 <!-- omit in toc -->

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4"},
            {"type": "text", "text": "Describe this video in detail. Follow the timeline and focus on on-screen text, interface changes, main actions, and scene changes."},
        ],
    }
]

downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
    downsample_mode=downsample_mode,
    max_num_frames=128,
    stack_frames=1,
    max_slice_nums=1,
    use_image_id=False,
).to(model.device)

generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

##### 高级参数 <!-- omit in toc -->

可以通过向 `apply_chat_template` 传递额外参数来自定义图片/视频处理行为：

| 参数 | 默认值 | 适用 | 说明 |
|------|--------|------|------|
| `downsample_mode` | `"16x"` | 图片 & 视频 | 视觉 token 下采样模式。`"16x"` 合并 token 以提高效率；`"4x"` 保留 4 倍 token 以获取更精细细节。**必须同时传给 `generate()`**。 |
| `max_slice_nums` | `9` | 图片 & 视频 | 高分辨率图片分片的最大数量。值越大可保留更多大图细节。推荐：图片 `36`，视频 `1`。 |
| `max_num_frames` | `128` | 仅视频 | 从视频中采样的最大主帧数。 |
| `stack_frames` | `1` | 仅视频 | 每秒的总采样点数。`1` = 仅主帧（不堆叠）。`N`（N>1）= 每秒 1 个主帧 + N−1 个子帧；子帧会被拼接为网格图并与主帧交替排列。推荐：`3` 或 `5`。 |
| `use_image_id` | `True` | 图片 & 视频 | 是否在每个图片/帧占位符前添加 `<image_id>N</image_id>` 标签。推荐：图片 `True`，视频 `False`。 |

> **注意：** `downsample_mode` 需要**同时**传给 `apply_chat_template`（确保占位符数量正确）和 `generate`（控制视觉编码器行为）。其他参数只需传给 `apply_chat_template`。

##### 使用 `transformers serve` 部署服务 <!-- omit in toc -->

Hugging Face Transformers 内置了一个轻量级 OpenAI 兼容服务器，适合快速测试和中等负载部署。

```bash
pip install "transformers[serving]>=5.7.0"
```

启动服务：

```bash
transformers serve openbmb/MiniCPM-V-4.6 --port 8000 --host 0.0.0.0 --continuous-batching
```

发送请求：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openbmb/MiniCPM-V-4.6",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"}},
        {"type": "text", "text": "What causes this phenomenon?"}
      ]
    }]
  }'
```

</details>


<a id="ios安卓鸿蒙端侧平台推理"></a>

#### iOS、安卓、鸿蒙端侧平台推理 <!-- omit in toc -->

我们将**全部端侧适配代码开源**，方便开发者在自己的设备上一键复现。可通过[下载页](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/DOWNLOAD_zh.md)体验，或参考[端侧部署仓库](https://github.com/OpenBMB/MiniCPM-V-Apps)获取完整源码。

#### 在其他训练、推理框架中使用 MiniCPM-V 4.6 <!-- omit in toc -->

MiniCPM-V 4.6 支持 SGLang, vLLM, llama.cpp, Ollama 等[推理框架](#训练和推理框架与-api-支持)，和 LLaMA-Factory, SWIFT 等[训练框架](#训练和推理框架与-api-支持)。

### 致谢 <!-- omit in toc -->

<details>
<summary>点击查看致谢。</summary>

我们对下列项目表示衷心感谢：

* [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) 提供了语言基座
* [SigLIP2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) 提供了视觉理解模块
* [Transformers](https://github.com/huggingface/transformers)

</details>

<a id="训练和推理框架与-api-支持"></a>

## 训练和推理框架与 API 支持

### 推理：vLLM、SGLang、llama.cpp、Ollama <!-- omit in toc -->

MiniCPM-V 4.6 推理适配 vLLM、SGLang、llama.cpp、Ollama 等框架。可点击下表查看部署指南，或参考我们的[使用指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook)。

| 框架 | MiniCPM-V 4.6 |
| :--- | :---: |
| vLLM | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_6_vllm_zh.md) |
| SGLang | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_6_sglang_zh.md) |
| llama.cpp | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_6_llamacpp_zh.md) |
| Ollama | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_6_ollama_zh.md) |

### 训练：LLaMA-Factory、SWIFT <!-- omit in toc -->

MiniCPM-V 4.6 支持通过 LLaMA-Factory 和 SWIFT 等框架训练。具体用法可以参考我们的[使用指南](https://github.com/OpenSQZ/MiniCPM-V-Cookbook)。

| 框架 | MiniCPM-V 4.6 |
| :--- | :---: |
| LLaMA-Factory | [微调指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/llamafactory_minicpmv46_zh.md) |
| SWIFT | [微调指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/swift_minicpmv46_zh.md) |

### API

MiniCPM-V 4.6 的 API 用法见 [API Guide](./api.md#minicpm-v-46)。

| 模型 | API |
|:---|:---:|
| MiniCPM-V 4.6 | [指南](./api.md#minicpm-v-46) |

## 模型库

| 模型               | 设备 | 资源 | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 简介          |                                                                                         下载链接                                                                                         |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 4.6 | GPU | 4 GB | MiniCPM-V 系列最小参数规模的端侧模型，以优秀的编码、解码效率完成单图、多图和视频理解任务。 |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6) |
| MiniCPM-V 4.6 gguf | CPU | 2 GB | gguf 版本，更低的内存占用和更高的推理效率。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-gguf) |
| MiniCPM-V 4.6 BNB | GPU | 3 GB | BNB（bitsandbytes int4）量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-BNB) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-BNB) |
| MiniCPM-V 4.6 AWQ  | GPU | 3 GB | AWQ 量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-AWQ) |
| MiniCPM-V 4.6 GPTQ | GPU | 3 GB | GPTQ 量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-GPTQ) |
| MiniCPM-V 4.6 Thinking | GPU | 4 GB | 思考模型版本，支持深度推理以应对更复杂的问题求解。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking) |
| MiniCPM-V 4.6 Thinking gguf | CPU | 2 GB | 思考模型的 gguf 版本，更低的内存占用和更高的推理效率。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-gguf) |
| MiniCPM-V 4.6 Thinking BNB | GPU | 3 GB | 思考模型的 BNB（bitsandbytes int4）量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-BNB) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-BNB) |
| MiniCPM-V 4.6 Thinking AWQ | GPU | 3 GB | 思考模型的 AWQ 量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-AWQ) |
| MiniCPM-V 4.6 Thinking GPTQ | GPU | 3 GB | 思考模型的 GPTQ 量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-GPTQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-GPTQ) |
