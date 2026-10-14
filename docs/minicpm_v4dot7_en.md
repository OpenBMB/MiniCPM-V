[← Home](../README.md) · [中文](./minicpm_v4dot7_zh.md)

# MiniCPM-V 4.7

[On-device results](#on-device-capability-and-efficiency) · [Evaluation](#evaluation) · [Examples](#examples) · [Usage](#usages) · [Model weights](#model-weights)

**MiniCPM-V 4.7** is the latest model in the MiniCPM-V series, available in **35BA3B and 1B** variants, with **35BA3B being the first MoE model in the MiniCPM-V series**. It builds on the family's single-image, multi-image, and video understanding capabilities, combining stronger foundation capabilities with efficient visual encoding and on-device inference. Notable features of MiniCPM-V 4.7 include:

- 📱 **The First Open-Source MoE Model with Lossless, Smooth On-Device Inference.**
  With our **[Lookahead Expert Scheduling (LES)](https://github.com/OpenBMB/llama.cpp-omni)** technique introduced alongside MiniCPM-V 4.7, combined with fine-grained layer-wise quantization and DSpark speculative sampling integration, MiniCPM-V 4.7 35BA3B becomes the **first open-source MoE model to run smoothly on-device without compromising performance**, achieving a decoding speed of **13+ tokens/s with just 3 GB of memory** on iPhone 17 Pro Max.
- 🔥 **A New Pareto Frontier in Performance and Efficiency.**
  As model capabilities continue to advance, inference efficiency has become an important measure of a model's value. On resource-constrained devices, delivering the strongest possible capabilities within a limited resource budget calls for higher capability density. The **35BA3B and 1B** variants of MiniCPM-V 4.7 each advance the performance–efficiency Pareto frontier for MLLMs. In Instruct mode, 35BA3B **achieves a higher overall score in less inference time than Qwen3.6 35BA3B**; the 1B variant continues to improve foundation capabilities over MiniCPM-V 4.6 and **substantially outperforms Qwen3.5 0.8B in both overall score and inference efficiency**.
- 🚀 **Efficient Architecture.**
  Based on the intra-ViT early compression technique from [LLaVA-UHD v4](https://github.com/THUMAI-Lab/LLaVA-UHD-v4), MiniCPM-V 4.7 compresses visual tokens early within the vision encoder, reducing computation in subsequent encoder layers. It also supports **mixed 4x/16x visual token compression**, allowing developers to balance visual detail and inference cost across tasks.
- 💻 **Broad Platform and Chip Support.**
  MiniCPM-V 4.7 supports **iOS, Android, and HarmonyOS**, together with open-source on-device deployment support. It also provides **day-0 support for chip platforms from Rockchip, Houmo, Intel, AMD, and Arm**, extending deployment options across mobile and local computing devices.
- 🛠️ **Developer Friendly.**
  MiniCPM-V 4.7 supports [inference frameworks](./deployment_en.md#supported-inference-and-training-frameworks-and-api) including SGLang, vLLM, llama.cpp, and Ollama, along with [fine-tuning frameworks](./deployment_en.md#supported-inference-and-training-frameworks-and-api) including SWIFT and LLaMA-Factory. It provides GGUF, BNB, AWQ, and GPTQ quantized variants and adds **FP8 quantization support**. We also release **[Self-Replay Fine-Tuning (SRFT)](https://openbmb.cn/en/blog/self-replay-fine-tuning)**, a fine-tuning approach that helps developers extend model capabilities at low cost while reducing forgetting of existing capabilities, making further customization easier.

### On-Device Capability and Efficiency

#### Smooth On-Device MoE Inference without compromising performance

An MoE model can have far more total parameters than it activates for a single token, but its on-device storage requirements are not limited to those active parameters. Different tokens can select different experts at different layers, while device memory often cannot hold all expert weights. Without a latency constraint, weights can remain on SSD or other storage, be loaded as needed, and be released after computation. This makes large-model inference possible, but repeated loading stalls can make decoding too slow for real-time interaction.

Another approach is to restrict the pool of available experts. For example, fixing an expert subset for an entire prompt can reduce loading frequency, but limits subsequent token-level routing to that subset, potentially excluding experts the model would otherwise select and reducing its capabilities. The central goal is therefore **smooth inference without compromising performance**: preserving model capability while delivering sufficient throughput under limited memory and bandwidth.

MiniCPM-V 4.7 35BA3B addresses this challenge through three complementary techniques:

- **[Lookahead Expert Scheduling (LES)](https://github.com/OpenBMB/llama.cpp-omni)** uses cross-layer routing prediction to preload experts likely to be needed by the next layer. Among the experts selected by the actual router, it prioritizes those successfully prefetched and overlaps their computation with loading the remaining required experts. Predictions guide only prefetching and scheduling; computation still follows the model's original token-level routing, and required experts missed by the prediction are loaded and evaluated as usual.
- **Fine-grained layer-wise quantization** reduces the storage footprint and memory traffic of expert weights while preserving model capability, making expert loading better suited to on-device memory and bandwidth constraints.
- **DSpark speculative sampling integration** incorporates speculative sampling into the on-device inference process, working with expert scheduling and quantization to improve decoding efficiency.

Together, these techniques make MiniCPM-V 4.7 35BA3B the first open-source MoE model to run smoothly on-device without compromising performance, achieving a decoding speed of **13+ tokens/s with just 3 GB of memory** on iPhone 17 Pro Max.

**On-Device MoE Inference: Before and After Optimization**

<p align="center">
  <a href="../assets/minicpmv4.7/moe_speed_comparison_en.gif"><img src="../assets/minicpmv4.7/moe_speed_comparison_en.gif" alt="On-device MoE inference speed comparison, with optimization on the left and without optimization on the right" width="95%"></a>
</p>

### Evaluation

**Significantly Advancing the Performance–Efficiency Pareto Frontier**

The 35BA3B and 1B variants of MiniCPM-V 4.7 each advance the performance–efficiency Pareto frontier. In Instruct mode, 35BA3B reaches a higher overall score in less inference time than Qwen3.6 35BA3B, and 1B substantially outperforms Qwen3.5 0.8B in both overall score and inference efficiency.

<p align="center">
  <a href="../assets/minicpmv4.7/performance_efficiency_frontier.png"><img src="../assets/minicpmv4.7/performance_efficiency_frontier.png" alt="Performance versus inference time for MiniCPM-V 4.7 35BA3B and 1B" width="95%"></a>
</p>

**Overall Performance (35BA3B, Instruct)**

**Vision-Language**

<p align="center">
  <a href="../assets/minicpmv4.7/overall_performance_35ba3b_instruct.png"><img src="../assets/minicpmv4.7/overall_performance_35ba3b_instruct.png" alt="Vision-Language performance comparison for MiniCPM-V 4.7 35BA3B in Instruct mode" width="90%"></a>
</p>

**Language**

<p align="center">
  <a href="../assets/minicpmv4.7/overall_performance_35ba3b_language.png"><img src="../assets/minicpmv4.7/overall_performance_35ba3b_language.png" alt="Language performance comparison for MiniCPM-V 4.7 35BA3B in Instruct mode" width="90%"></a>
</p>

**Overall Performance (1B, Instruct)**

<p align="center">
  <a href="../assets/minicpmv4.7/overall_performance_1b_instruct.png"><img src="../assets/minicpmv4.7/overall_performance_1b_instruct.png" alt="Full performance comparison for MiniCPM-V 4.7 1B in Instruct mode" width="90%"></a>
</p>

### Examples

MiniCPM-V 4.7 can run on phones and tablets. The clips below are unedited screen recordings captured directly on the devices.

<table align="center">
  <tr>
    <td align="center"><b>35BA3B</b><br><sub>iPhone 17 Pro Max</sub></td>
    <td align="center"><b>35BA3B</b><br><sub>iPad M4</sub></td>
    <td align="center"><b>1B</b><br><sub>iPhone 17 Pro Max</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="../assets/minicpmv4.7/v47_35ba3b_iphone2_en.gif" height="520"/></td>
    <td align="center"><img src="../assets/minicpmv4.7/v47_35ba3b_ipad_en.gif" height="520"/></td>
    <td align="center"><img src="../assets/minicpmv4.7/v47_1b_iphone_en.gif" height="520"/></td>
  </tr>
</table>

### Usages

#### Inference with Transformers <!-- omit in toc -->
<details>
<summary>Click to expand the Transformers inference examples.</summary>

##### Installation <!-- omit in toc -->

```bash
pip install "transformers[torch]>=5.18.0" torchvision torchcodec
```

> **Note on CUDA compatibility:** `torchcodec` (used for video decoding) may have compatibility issues with certain CUDA versions. For example, `torch>=2.11` bundles CUDA 13.1 by default, while environments with CUDA 12.x may encounter errors such as `RuntimeError: Could not load libtorchcodec`. Two workarounds:
>
> 1. **Replace `torchcodec` with `PyAV`** — supports both image and video inference without CUDA version constraints:
>    ```bash
>    pip install "transformers[torch]>=5.18.0" torchvision av
>    ```
> 2. **Pin the CUDA version** when installing torch to match your environment (e.g. CUDA 12.8):
>    ```bash
>    pip install "transformers>=5.18.0" torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
>    ```

##### Load the Model <!-- omit in toc -->

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# Flash Attention 2 is recommended for faster inference and lower GPU memory usage,
# especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
```

##### Image Inference <!-- omit in toc -->

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
    enable_thinking=False,
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

##### Video Inference <!-- omit in toc -->

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
    enable_thinking=False,
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

##### Advanced Parameters <!-- omit in toc -->

You can customize image/video processing by passing additional parameters to `apply_chat_template`:

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `downsample_mode` | `"16x"` | Image & Video | Visual token downsampling. `"16x"` merges tokens for efficiency; `"4x"` keeps four times as many tokens for finer detail. **Must also be passed to `generate()`**. |
| `max_slice_nums` | `9` | Image & Video | Maximum number of slices when splitting a high-resolution image. Higher values preserve more detail for large images. Recommended: `36` for image, `1` for video. |
| `max_num_frames` | `128` | Video only | Maximum number of main frames sampled from the video. |
| `stack_frames` | `1` | Video only | Total sample points per second. `1` = main frame only (no stacking). `N` (N>1) = 1 main frame + N−1 sub-frames per second; the sub-frames are composited into a grid image and interleaved with main frames. Recommended: `3` or `5`. |
| `use_image_id` | `True` | Image & Video | Whether to prepend `<image_id>N</image_id>` tags before each image/frame placeholder. Recommended: `True` for image, `False` for video. |

> **Note:** `downsample_mode` must be passed to **both** `apply_chat_template` (for correct placeholder count) and `generate` (for the vision encoder). All other parameters only need to be passed to `apply_chat_template`.

##### Serving with `transformers serve` <!-- omit in toc -->

Hugging Face Transformers includes a lightweight OpenAI-compatible server for quick testing and moderate-load deployment.

```bash
pip install "transformers[serving]>=5.18.0"
```

Start the server:

```bash
transformers serve openbmb/MiniCPM-V-4.7-35B-A3B --port 8000 --host 0.0.0.0
```

Send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"}},
        {"type": "text", "text": "What causes this phenomenon?"}
      ]
    }]
  }'
```

Tool calling example:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "the weather of Beijing"}
  ]}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }
  }]
}'
```

##### Recommended Sampling Parameters <!-- omit in toc -->

We recommend using the following set of sampling parameters for generation in instruct mode: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.

</details>

#### Speculative decoding (DSpark) <!-- omit in toc -->
<details>
<summary>Click to show speculative decoding. </summary>

Speculative decoding (DSpark): we also release [MiniCPM-V-4.7-35B-A3B-DSpark](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark)/[MiniCPM-V-4.7-1B-DSpark](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-DSpark), a DSpark draft model trained for [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)/[MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B).

</details>

#### Inference with OpenVINO™

[OpenVINO™](https://docs.openvino.ai) is an open-source toolkit for optimizing and deploying AI inference on Intel CPU/GPU/NPU platforms. OpenVINO provides day-0 support for MiniCPM-V-4.7 on Intel platforms with high-performance inference. See the [OpenVINO™ Toolkit](https://github.com/openvinotoolkit) to get started. You can install and run the example with [this notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-4.7).

#### On-Device Deployment

MiniCPM-V 4.7 supports deployment on iOS, Android, and HarmonyOS, with day-0 support for Rockchip, Houmo, Intel, AMD, and Arm chip platforms. See the [edge deployment repository](https://github.com/OpenBMB/MiniCPM-V-Apps) for the apps and source code.

#### Inference and Fine-Tuning

For supported inference and training frameworks, see [Supported Inference and Training Frameworks and API](./deployment_en.md#supported-inference-and-training-frameworks-and-api). The [MiniCPM-V 4.7 Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/docs/pages/v4.7/overview.md) covers deployment, fine-tuning, and quantization workflows.

**Task guides**

- [Single-image chat](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_7_single_image.md)
- [Multi-image chat](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_7_multi_images.md)
- [Video understanding](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_7_video_understanding.md)
- [OCR](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_7_ocr.md)
- [PDF parsing](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_7_pdf_parse.md)
- [Visual grounding](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_7_grounding.md)

**Quantization**

| Method | Guide |
|:---|:---|
| GGUF | [Quantization guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/gguf/minicpm-v4_7_gguf_quantize.md) |
| BNB | [Quantization guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/bnb/minicpm-v4_7_bnb_quantize.md) |
| AWQ | [Quantization guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/awq/minicpm-v4_7_awq_quantize.md) |
| GPTQ | [Quantization guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/gptq/minicpm-v4_7_gptq_quantize.md) |
| FP8 / INT4 | [Quantization guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/llmcompressor/minicpm-v4_7_llmcompressor_quantize.md) |

### Acknowledgements

<details>
<summary>Click to view acknowledgements.</summary>

We would like to thank the following projects:

* [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) and [Qwen3.6](https://huggingface.co/collections/Qwen/qwen36) for providing the language backbone
* [SigLIP2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) for providing the vision understanding module
* [Transformers](https://github.com/huggingface/transformers)

</details>

## Model Weights

MiniCPM-V 4.7 is available in the following variants:

| Model | Description | Download |
|:---|:---|:---:|
| MiniCPM-V 4.7 35BA3B | The first open-source MoE model to run smoothly on-device without compromising performance, raising the bar for on-device performance. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B) |
| MiniCPM-V 4.7 35BA3B FP8 | FP8 quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-FP8) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-FP8) |
| MiniCPM-V 4.7 35BA3B int4 | int4 quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-int4) |
| MiniCPM-V 4.7 35BA3B AWQ | AWQ (W4A16) quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-AWQ) |
| MiniCPM-V 4.7 35BA3B GPTQ | GPTQ (W4A16) quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-GPTQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-GPTQ) |
| MiniCPM-V 4.7 35BA3B GGUF | GGUF version with lower memory usage for CPU inference. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf) |
| MiniCPM-V 4.7 35BA3B DSpark | DSpark draft model for speculative decoding. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark) |
| MiniCPM-V 4.7 1B | Combines stronger capabilities with extremely high efficiency, advancing the performance–efficiency Pareto frontier for multimodal large language models. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B) |
| MiniCPM-V 4.7 1B int4 | bitsandbytes int4 quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-int4) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-int4) |
| MiniCPM-V 4.7 1B AWQ | AWQ (W4A16) quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-AWQ) |
| MiniCPM-V 4.7 1B GPTQ | GPTQ (W4A16) quantized version with lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-GPTQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-GPTQ) |
| MiniCPM-V 4.7 1B GGUF | GGUF version with lower memory usage for CPU inference. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-gguf) |
| MiniCPM-V 4.7 1B DSpark | DSpark draft model for speculative decoding. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-DSpark) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-DSpark) |

Also available on [Ollama](https://ollama.com/openbmb/minicpm-v4.7).
