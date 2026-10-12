## MiniCPM-V 4.6

> Archived on: 2026-10-10

[English](./minicpm_v4dot6_en.md) | [简体中文](./minicpm_v4dot6_zh.md) | [Back to README](../README.md)

[Online Demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-4.6-Demo)


**MiniCPM-V 4.6** is our most edge-deployment-friendly model to date. The model is built based on SigLIP2-400M and the Qwen3.5-0.8B LLM. It inherits the strong single-image, multi-image, and video understanding capabilities of MiniCPM-V family, while significantly improving computation efficiency. It also introduces mixed 4x/16x visual token compression. Notable features of MiniCPM-V 4.6 include:

- 🔥 **Leading Foundation Capability.**
  MiniCPM-V 4.6 scores 13 on the Artificial Analysis Intelligence Index benchmark, outperforming Qwen3.5-0.8B's score of 10 with 19x fewer token cost, and Qwen3.5-0.8B-Thinking's score of 11 with 43x fewer token cost. It also surpasses the larger Ministral 3 3B (score of 11).

- 💪 **Strong Multimodal Capability.**
  MiniCPM-V 4.6 outperforms Qwen3.5-0.8B on most vision-language understanding tasks, and reaches Qwen3.5 2B-level capability on many benchmarks including OpenCompass, RefCOCO, HallusionBench, MUIRBench, and OCRBench.
- 🚀 **Ultra-Efficient Architecture.**
  Based on the latest technique in [LLaVA-UHD v4](https://huggingface.co/papers/2605.08985), MiniCPM-V 4.6 reduces the visual encoding computation FLOPs by more than 50%. It enables MiniCPM-V 4.6 to achieve better efficiency to even smaller models, achieving ~1.5x token throughput compared to Qwen3.5-0.8B.
  It also supports mixed 4x/16x visual token compression rate, allowing flexible switching between accuracy and speed.
- 📱 **Broad Mobile Platform Coverage.**
  MiniCPM-V 4.6 can be deployed across all three mainstream mobile platforms — iOS, Android, and HarmonyOS. With every edge adaptation code open-sourced, developers can reproduce the on-device experience in [just a few steps](#deploy-minicpm-v-46-on-ios-android-and-harmonyos-platforms).
- 🛠️ **Developer Friendly.**
  MiniCPM-V 4.6 is adapted to [inference frameworks](#supported-inference-and-training-frameworks-and-api) such as SGLang, vLLM, llama.cpp, Ollama, and supports [fine-tuning ecosystems](#supported-inference-and-training-frameworks-and-api) such as SWIFT and LLaMA-Factory. Developers can quickly customize models for new domains and tasks on consumer-grade GPUs. We provide multiple quantized variants across GGUF, BNB, AWQ, and GPTQ formats.


### Evaluation <!-- omit in toc -->

**Overall Performance (Instruct)**

<p align="center">
  <img src="../assets/minicpmv4.6/instruct.png" width="90%"></img>
</p>


<details>
<summary>Click to view MiniCPM-V 4.6-Thinking performance.</summary>


<p align="center">
  <img src="../assets/minicpmv4.6/thinking.png" width="90%"></img>
</p>


</details>

<br>

**MiniCPM-V 4.6 Inference Efficiency**
<table align="center">
  <tr>
    <td align="center" width="50%"><b>High-Concurrency Throughput</b></td>
    <td align="center" width="50%"><b>Single Request TTFT (ms)</b></td>
  </tr>
  <tr>
    <td align="center" valign="middle"><img src="../assets/minicpmv4.6/throughput.png" width="110%"></img></td>
    <td align="center" valign="middle"><img src="../assets/minicpmv4.6/ttft.png" width="100%"></img></td>
  </tr>
</table>


### Examples <!-- omit in toc -->

<div align="center">
  <a href="https://www.youtube.com/watch?v=Ch5UG1FoysM"><img src="../assets/minicpmv4.6/video_play.png" width="70%"></a>
</div>

MiniCPM-V 4.6 can be deployed across three mainstream end-side platforms — **iOS, Android and HarmonyOS**. The clips below are raw screen recordings on phone devices without edition.

<table align="center">
  <tr>
    <td align="center"><b>iPhone</b><br><sub>iPhone 17 Pro Max</sub></td>
    <td align="center"><b>Android</b><br><sub>Redmi K70</sub></td>
    <td align="center"><b>HarmonyOS</b><br><sub>HUAWEI nova 14</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="../assets/minicpmv4.6/v46_iphone_en_handwriting.gif" width="100%"/></td>
    <td align="center"><img src="../assets/minicpmv4.6/v46_android_en_refraction.gif" width="100%"/></td>
    <td align="center"><img src="../assets/minicpmv4.6/v46_harmonyos_en_ticket.gif" width="100%"/></td>
  </tr>
</table>


### Usages

#### Inference with Transformers <!-- omit in toc -->
<details>
<summary>Click to show inference examples with Transformers. </summary>

##### Installation <!-- omit in toc -->

```bash
pip install "transformers[torch]>=5.7.0" torchvision torchcodec
```

> **Note on CUDA compatibility:** `torchcodec` (used for video decoding) may have compatibility issues with certain CUDA versions. For example, `torch>=2.11` bundles CUDA 13.1 by default, while environments with CUDA 12.x may encounter errors such as `RuntimeError: Could not load libtorchcodec`. Two workarounds:
>
> 1. **Replace `torchcodec` with `PyAV`** — supports both image and video inference without CUDA version constraints:
>    ```bash
>    pip install "transformers[torch]>=5.7.0" torchvision av
>    ```
> 2. **Pin the CUDA version** when installing torch to match your environment (e.g. CUDA 12.8):
>    ```bash
>    pip install "transformers>=5.7.0" torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
>    ```

##### Load Model <!-- omit in toc -->

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4.6"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# Flash Attention 2 is recommended for better acceleration and memory saving,
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
| `downsample_mode` | `"16x"` | Image & Video | Visual token downsampling. `"16x"` merges tokens for efficiency; `"4x"` keeps 4× more tokens for finer detail. Must also be passed to `generate()`. |
| `max_slice_nums` | `9` | Image & Video | Maximum number of slices when splitting a high-resolution image. Higher values preserve more detail for large images. Recommended: `36` for image, `1` for video. |
| `max_num_frames` | `128` | Video only | Maximum number of main frames sampled from the video. |
| `stack_frames` | `1` | Video only | Total sample points per second. `1` = main frame only (no stacking). `N` (N>1) = 1 main frame + N−1 sub-frames per second; the sub-frames are composited into a grid image and interleaved with main frames. Recommended: `3` or `5`. |
| `use_image_id` | `True` | Image & Video | Whether to prepend `<image_id>N</image_id>` tags before each image/frame placeholder. Recommended: `True` for image, `False` for video. |

> **Note:** `downsample_mode` must be passed to **both** `apply_chat_template` (for correct placeholder count) and `generate` (for the vision encoder). All other parameters only need to be passed to `apply_chat_template`.

##### Serving with `transformers serve` <!-- omit in toc -->

Hugging Face Transformers includes a lightweight OpenAI-compatible server for quick testing and moderate-load deployment.

```bash
pip install "transformers[serving]>=5.7.0"
```

Start the server:

```bash
transformers serve openbmb/MiniCPM-V-4.6 --port 8000 --host 0.0.0.0 --continuous-batching
```

Send a request:

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



<a id="deploy-minicpm-v-46-on-ios-android-and-harmonyos-platforms"></a>

#### Deploy MiniCPM-V 4.6 on iOS, Android, and HarmonyOS Platforms <!-- omit in toc -->

We **open-source edge deployment guidance** of all these platforms, so that developers can deploy on their own devices in a few steps. Try the apps via the [download page](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/DOWNLOAD.md), or follow the [edge deployment guide](https://github.com/OpenBMB/MiniCPM-V-Apps) for the full source.

#### Use MiniCPM-V 4.6 in Other Inference and Training Frameworks <!-- omit in toc -->

MiniCPM-V 4.6 supports [inference frameworks](#supported-inference-and-training-frameworks-and-api) including SGLang, vLLM, llama.cpp, Ollama, and [training frameworks](#supported-inference-and-training-frameworks-and-api) including LLaMA-Factory, SWIFT.


### Acknowledgements <!-- omit in toc -->

<details>
<summary>Click to view acknowledgements.</summary>

We would like to thank the following projects:
* [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) for providing language backbone
* [SigLIP2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) for providing vision understanding module
* [Transformers](https://github.com/huggingface/transformers)

</details>

<a id="supported-inference-and-training-frameworks-and-api"></a>

## Supported Inference and Training Frameworks and API

### Inference: vLLM, SGLang, llama.cpp, Ollama <!-- omit in toc -->

MiniCPM-V 4.6 supports inference with vLLM, SGLang, llama.cpp and Ollama. See the deployment guides below, or refer to our [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) for more details.

| Framework | MiniCPM-V 4.6 |
| :--- | :---: |
| vLLM | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_6_vllm.md) |
| SGLang | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_6_sglang.md) |
| llama.cpp | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_6_llamacpp.md) |
| Ollama | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_6_ollama.md) |

### Training: LLaMA-Factory, SWIFT <!-- omit in toc -->

MiniCPM-V 4.6 supports fine-tuning with LLaMA-Factory and SWIFT. Refer to our [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-Cookbook) for more details.

| Framework | MiniCPM-V 4.6 |
| :--- | :---: |
| LLaMA-Factory | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/llamafactory_minicpmv46.md) |
| SWIFT | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/swift_minicpmv46.md) |

### API

Refer to the [API Guide](./api.md#minicpm-v-46) for MiniCPM-V 4.6.

| Model | API |
|:---|:---:|
| MiniCPM-V 4.6 | [Guide](./api.md#minicpm-v-46) |

## Model Zoo

| Model           | Device | Memory    | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Description       | Download |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 4.6 | GPU | 4 GB | The smallest MiniCPM-V model to date, with strong encoding and decoding efficiency for single-image, multi-image, and video understanding tasks. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6) |
| MiniCPM-V 4.6 gguf | CPU | 2 GB | The gguf version, lower memory usage and faster inference. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-gguf) |
| MiniCPM-V 4.6 BNB | GPU | 3 GB | The BNB (bitsandbytes int4) quantized version, lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-BNB) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-BNB) |
| MiniCPM-V 4.6 AWQ | GPU | 3 GB | The AWQ quantized version, lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-AWQ) |
| MiniCPM-V 4.6 GPTQ | GPU | 3 GB | The GPTQ quantized version, lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-GPTQ) |
| MiniCPM-V 4.6 Thinking | GPU | 4 GB | The thinking variant, supporting deep reasoning for more complex problem solving. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking) |
| MiniCPM-V 4.6 Thinking gguf | CPU | 2 GB | The gguf version of the thinking variant, lower memory usage and faster inference. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-gguf) |
| MiniCPM-V 4.6 Thinking BNB | GPU | 3 GB | The BNB (bitsandbytes int4) quantized version of the thinking variant, lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-BNB) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-BNB) |
| MiniCPM-V 4.6 Thinking AWQ | GPU | 3 GB | The AWQ quantized version of the thinking variant, lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-AWQ) |
| MiniCPM-V 4.6 Thinking GPTQ | GPU | 3 GB | The GPTQ quantized version of the thinking variant, lower GPU memory usage. | [🤗](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking-GPTQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking-GPTQ) |
