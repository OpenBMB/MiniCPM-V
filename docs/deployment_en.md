[← Home](../README.md) · [中文](./deployment_zh.md)

# Deployment, Training, and API

Run the code examples from the repository root unless stated otherwise.

## MiniCPM-V & o Cookbook

The [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) provides scenario-based recipes for deploying, fine-tuning, quantizing, and building demos with MiniCPM-V and MiniCPM-o. The [documentation website](https://minicpm-o.readthedocs.io/en/latest/index.html) presents these recipes in a structured way for quick lookup.

For MiniCPM-o 4.5 API usage, see the [API Guide](./api.md). Previous MiniCPM-V 4.6 usage is available in its [archived documentation](./minicpm_v4dot6_en.md).

It is organized for:

* **Individuals**: Local inference, quantized deployment, and end-device demos.
* **Enterprises**: Scalable serving, high-throughput inference, and production-oriented deployment.
* **Researchers**: Fine-tuning, model adaptation, and experimental workflows.

For framework-specific deployment and training guides, see [Supported Inference and Training Frameworks and API](./deployment_en.md#supported-inference-and-training-frameworks-and-api).


## Supported Inference and Training Frameworks and API

### Inference: vLLM, SGLang, llama.cpp, Ollama, FlagOS

We support inference with vLLM, SGLang, llama.cpp and Ollama. See the per-model deployment guides below, or refer to our [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) for more details.

| Framework | MiniCPM-V 4.7 | MiniCPM-o 4.5 | MiniCPM-V 4.5 | MiniCPM-V 4.0 | Previous MiniCPM-V/o Models |
|:---|:---:|:---:|:---:|:---:|:---:|
| vLLM | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_7_vllm.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-o4_5_vllm.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_5_vllm.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_vllm.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/README.md) |
| SGLang | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_7_sglang.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/MiniCPM-o4_5_sglang.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_5_sglang.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/MiniCPM-v4_sglang.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/README.md) |
| llama.cpp | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_7_llamacpp.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-o4_5_llamacpp.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_5_llamacpp.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_llamacpp.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/README.md) |
| Ollama | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_7_ollama.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-o4_5_ollama.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_5_ollama.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_ollama.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/README.md) |

#### FlagOS

FlagOS supports inference of MiniCPM-o 4.5 across six AI chip families, including Nvidia GPUs.

<details>
<summary>Click to show FlagOS support details.</summary>

To enable large-scale deployment across different AI chips, Beijing Zhiyuan Research Institute, together with numerous research institutions, chip manufacturers, system vendors, and algorithm and software organizations both domestically and internationally, jointly initiated and established the FlagOS Open Source Community.

The FlagOS community is dedicated to building a unified, open-source system software stack for various AI chips, encompassing core open-source projects such as a large-scale operator library, a unified AI compiler, parallel training and inference frameworks, and a unified communication library. It aims to create an open technology ecosystem connecting the “model-system-chip” layers. By enabling “develop once, deploy across chips”, FlagOS unlocks the computational potential of hardware, breaks down the ecosystem silos between different chip software stacks, and effectively reduces migration costs for developers. The FlagOS community fosters an AI hardware and software ecosystem, overcomes single-vendor closed-source monopolies, promotes widespread deployment of AI hardware technologies, and is committed to rooted in China while embracing global collaboration.
Official website: https://flagos.io.

##### FlagOS: Supporting Multiple AI Chips

Thanks to FlagOS’s unified multi-chip AI system software stack, MiniCPM-o 4.5 was adapted to 6 different AI chips in an extremely short time. Currently, the multi-chip version of MiniCPM-o 4.5 has been released on FlagRelease, FlagOS’s platform for automatic migration, adaptation, and deployment of large models across multi-architecture AI chips. Details are as follows:

| Vendor          | ModelScope   | Huggingface  |
|:----------------|:------------:|:------------:|
| Nvidia          | [MiniCPM-o-4.5-nvidia-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) | [MiniCPM-o-4.5-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) |
| Hygon-BW1000    | [MiniCPM-o-4.5-hygon-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) | [MiniCPM-o-4.5-hygon-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) |
| Metax-C550      | [MiniCPM-o-4.5-metax-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) | [MiniCPM-o-4.5-metax-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) |
| Iluvatar-BIV150 | [MiniCPM-o-4.5-iluvatar-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) | [MiniCPM-o-4.5-iluvatar-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) |
| Ascend-A3       | [MiniCPM-o-4.5-ascend-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) | [MiniCPM-o-4.5-ascend-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) |
| Zhenwu-810E     | [MiniCPM-o-4.5-zhenwu-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) | [MiniCPM-o-4.5-zhenwu-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) |

###### Comprehensive Evaluation

**Transformers-FlagOS version**

Accuracy Difference between `USE_FLAGOS=1` on multi-backend and `USE_FLAGOS=0` on Nvidia-CUDA

| Metrics                  | FlagOS Backend  | Difference with Nvidia-CUDA |
|:-------------------------|:---------------:|:---------------------------:|
| Video-MME 0-shot avg@1 ↑ |     Nvidia      |            0.33%            |
| Video-MME 0-shot avg@1 ↑ |  Hygon-BW1000   |            0.17%            |
| Video-MME 0-shot avg@1 ↑ |    Ascend-A3    |            0.50%            |
| Video-MME 0-shot avg@1 ↑ | Iluvatar-BIV150 |            1.83%            |
| Video-MME 0-shot avg@1 ↑ |   Metax-C550    |            0.75%            |


**VLLM-FlagOS version**

Accuracy Difference between `USE_FLAGGEMS=1 FLAGCX_PATH=/workspace/FlagCX` on Nvidia or `USE_FLAGGEMS=1` on ZHENW 810E, and launching vllm server directly on Nvidia

| Metrics (avg@1)     | Difference between Nvidia-FlagOS and Nvidia-CUDA | Difference between Zhenwu-FlagOS and Nvidia-CUDA |
|:--------------------|:------------------------------------------------:|:------------------------------------------------:|
| CMMMU ↑             | 0.72% | 3.5% |
| MMMU ↑              | 1.44% | 1.18% |
| MMMU_Pro_standard ↑ | 0.83% | 0.22% |
| MM-Vet v2 ↑         | 0.46% | 1.33% |
| OCRBench ↑          | 0.10% | 1% |
| CII-Bench ↑         | 0.40% | 0.13% |
| Blink ↑             | 1.90% | 2.19% |

##### FlagOS Usage

###### FlagOS Performance Acceleration on Nvidia

On the Transformers version, under the premise of precision alignment between the CUDA and FlagOS ecosystems, FlagOS achieves a 6% performance improvement in total task execution time compared to CUDA.

**From FlagRelease【Recommendation】**

FlagRelease is a platform developed by the FlagOS team for automatic migration, adaptation, and deployment of large models across multi-architecture AI chips. The multi-chip version of MiniCPM-o 4.5 has already been released on FlagRelease. All necessary software packages are pre-installed on the platform, so users do not need to install anything.

- FlagRelease Image Key Versions  

  | Component               | Version                             |
  |:------------------------|:------------------------------------|
  | Accelerator Card Driver | 570.158.01                          |
  | CUDA SDK Build          | cuda_13.0.r13.0/compiler.36424714_0 |
  | FlagTree                | 0.4.0+3.5                           |
  | FlagGems                | 4.2.1rc0                            |
  | vllm & vllm-plugin-fl   | 0.13.0 + vllm_fl 0.0.0              |
  | FlagCX                  | 0.1.0                               |  

- FlagRelease Quick Start

  | Vendor     | ModelScope   | Huggingface  |
  |:-----------|:------------:|:------------:|
  | Nvidia | [MiniCPM-o-4.5-nvidia-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) | [MiniCPM-o-4.5-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) |
  | Hygon-BW1000 | [MiniCPM-o-4.5-hygon-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) | [MiniCPM-o-4.5-hygon-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) |
  | Metax-C550 | [MiniCPM-o-4.5-metax-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) | [MiniCPM-o-4.5-metax-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) |
  | Iluvatar-BIV150 | [MiniCPM-o-4.5-iluvatar-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) | [MiniCPM-o-4.5-iluvatar-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) |
  | Ascend-A3 | [MiniCPM-o-4.5-ascend-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) | [MiniCPM-o-4.5-ascend-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) |
  | Zhenwu-810E | [MiniCPM-o-4.5-zhenwu-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) | [MiniCPM-o-4.5-zhenwu-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) |  


###### From Scratch

- Dependencies: Python 3.12, GLIBC 2.39, GLIBCXX 3.4.33, CXXABI 1.3.15

**Transformers**

- Installing the FlagOS Operator Library

  Official Repository: https://github.com/flagos-ai/FlagGems
  
  ```shell
  pip install flag-gems==4.2.1rc0
  ```

- Installing the FlagOS Compiler

  Official Repository: https://github.com/flagos-ai/flagtree
  
  Quick Reference for Core Dependency Versions: https://github.com/flagos-ai/FlagTree/blob/main/documents/build.md#tips-for-building
  
  ```shell
  pip uninstall triton
  
  python3 -m pip install flagtree==0.4.0+3.5 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net
  ```

- Activating Acceleration

  Add `USE_FLAGOS=1` before the command for the task you want to run. For example, when you run:
  ```shell
  python3 generate_speech_from_video.py
  ```
  
  To use the MiniCPM-o-4.5 model to generate spoken responses from video content, you can:
  
  ```shell
  USE_FLAGOS=1 python3 generate_speech_from_video.py
  ```
  
  to accelerate this process with FlagOS.

**Vllm Version**

- Installing the FlagOS Operator Library

  Official Repository: https://github.com/flagos-ai/FlagGems

  ```shell
  pip install flag-gems==4.2.1rc0
  pip install triton==3.5.1
  ```

- Activating Acceleration

  Add `USE_FLAGOS=1` before the command for the task you want to run. For example, when you run:
  ```shell
  vllm serve ${model_path} --dtype auto  --gpu_memory_utilization 0.9 --trust-remote-code --max-num-batched-tokens 2048 --served-model-name cpmo --port ${Port}
  ```

  To start the MiniCPM-o-4.5 server, you can:
  ```shell
  USE_FLAGOS=1 vllm serve ${model_path} --dtype auto  --gpu_memory_utilization 0.9 --trust-remote-code --max-num-batched-tokens 2048 --served-model-name cpmo --port ${Port}
  ```
  to accelerate this process with FlagOS.

##### Using FlagOS Unified Multi-Chip Backend Plugin

[vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) is a plugin built for the vLLM inference/service framework. Developed on top of FlagOS’s unified multi-chip backend, it is designed to extend vLLM’s capabilities and performance across a variety of hardware environments.

###### Using vllm-plugin-FL

| Vendor | From Scratch | From FlagRelease |
|:-------|:-------------|:----------------|
| Nvidia | [vllm-plugin-FL/MiniCPM-o-4.5](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/minicpm/README.md) | [MiniCPM-o-4.5-ModelScope](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS), [MiniCPM-o-4.5-HuggingFace](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) |

</details>

### Training: LLaMA-Factory, SWIFT

We support fine-tuning with LLaMA-Factory, SWIFT. Refer to our [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-Cookbook) for more details.

| Framework | MiniCPM-V 4.7 | Previous MiniCPM-V/o Models |
|:---|:---:|:---:|
| LLaMA-Factory | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/tree/main/finetune/llamafactory) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/tree/main/finetune/llamafactory) |
| SWIFT | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/swift/swift_minicpmv47_35b.md) | [Guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/tree/main/finetune/swift) |


### API

Refer to the [API Guide](./api.md) for MiniCPM-V 4.7 / 4.5 and MiniCPM-o 4.5. The MiniCPM-V 4.6 API entry is preserved in its [archived documentation](./minicpm_v4dot6_en.md#api).

| Model | API |
|:---|:---:|
| MiniCPM-V 4.7 | [Guide](./api.md#minicpm-v-47) |
| MiniCPM-V 4.5 | [Guide](./api.md#minicpm-v-45) |
| MiniCPM-o 4.5 | [Guide](./api.md#minicpm-o-45) |
