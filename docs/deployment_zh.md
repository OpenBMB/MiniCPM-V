[← 首页](../README_zh.md) · [English](./deployment_en.md)

# 部署、训练与 API

除另有说明外，以下代码均从仓库根目录运行。

## MiniCPM-V & o 使用手册

欢迎探索我们整理的 [MiniCPM-V & o 使用手册 (Cookbook)](https://github.com/OpenSQZ/MiniCPM-V-CookBook)。Cookbook 提供面向场景的教程，覆盖 MiniCPM-V 和 MiniCPM-o 的部署、微调、量化和 Demo 构建等常见任务；配套的[文档网站](https://minicpm-o.readthedocs.io/en/latest/index.html)以结构化方式呈现这些方案，便于快速查找。

MiniCPM-o 4.5 的 API 用法见 [API Guide](./api.md)。历史版本 MiniCPM-V 4.6 的用法见[独立文档](./minicpm_v4dot6_zh.md)。

它面向以下使用场景组织内容：

* **个人用户**：本地推理、量化部署和端侧 Demo。
* **企业用户**：可扩展服务、高吞吐推理和生产部署。
* **研究者**：微调、模型适配和实验工作流。

如需查看具体框架的部署和训练指南，请参考[训练和推理框架与 API 支持](./deployment_zh.md#训练和推理框架与-api-支持)。


## 训练和推理框架与 API 支持

### 推理：vLLM、SGLang、llama.cpp、Ollama、FlagOS

MiniCPM-V 和 MiniCPM-o 模型推理适配 vLLM、SGLang、llama.cpp、Ollama 等框架。可点击下表查看各模型的部署指南，或参考我们的[使用指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook)。

| 框架 | MiniCPM-V 4.7 | MiniCPM-o 4.5 | MiniCPM-V 4.5 | MiniCPM-V 4.0 | 更多 MiniCPM-V/o 模型 |
|:---|:---:|:---:|:---:|:---:|:---:|
| vLLM | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_7_vllm_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-o4_5_vllm_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_5_vllm_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_vllm_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/README.md) |
| SGLang | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_7_sglang_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-o4_5_sglang_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_5_sglang_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_sglang_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/README.md) |
| llama.cpp | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_7_llamacpp_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-o4_5_llamacpp_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_5_llamacpp_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_llamacpp_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/README.md) |
| Ollama | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_7_ollama_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-o4_5_ollama_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_5_ollama_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_ollama_zh.md) | [部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/README.md) |

#### FlagOS

FlagOS 平台支持 MiniCPM-o 4.5 在6 种不同的 AI 芯片上的推理，包括 Nvidia GPU 等。

<details>
<summary>点击展开 FlagOS 上的 MiniCPM-o 4.5 推理详细</summary>

为解决不同 AI 芯片大规模落地应用，北京智源研究院联合众多科研机构、芯片企业、系统厂商、算法和软件相关单位等国内外机构共同发起并创立了 FlagOS 开源社区。

FlagOS 社区致力于打造面向多种 AI 芯片的统一、开源的系统软件栈，包括大型算子库、统一AI编译器、并行训推框架、统一通信库等核心开源项目，构建「模型-系统-芯片」三层贯通的开放技术生态，通过“一次开发跨芯迁移”释放硬件计算潜力，打破不同芯片软件栈之间生态隔离，有效降低开发者的迁移成本。FlagOS 社区构建人工智能软硬件生态，突破单一闭源垄断，推动AI硬件技术大范围落地发展，立足中国、拥抱全球合作。
官网速递：https://flagos.io

##### FlagOS 多 AI 芯片支持

基于FlagOS极短时间内适配MiniCPM-o 4.5到 6 种不同的 AI 芯片，得益于众智 FlagOS 的多芯片统一 AI 系统软件栈的能力。目前，在FlagOS团队构建的面向多架构人工智能芯片的大模型自动迁移、适配与发布平台FlagRelease上，已发布MiniCPM-o-4.5的多芯片版本。细节如下：

| Vendor          | ModelScope   | Huggingface  |
|:----------------|:------------:|:------------:|
| Nvidia          | [MiniCPM-o-4.5-nvidia-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) | [MiniCPM-o-4.5-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) |
| Hygon-BW1000    | [MiniCPM-o-4.5-hygon-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) | [MiniCPM-o-4.5-hygon-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) |
| Metax-C550      | [MiniCPM-o-4.5-metax-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) | [MiniCPM-o-4.5-metax-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) |
| Iluvatar-BIV150 | [MiniCPM-o-4.5-iluvatar-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) | [MiniCPM-o-4.5-iluvatar-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) |
| Ascend-A3       | [MiniCPM-o-4.5-ascend-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) | [MiniCPM-o-4.5-ascend-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) |
| Zhenwu-810E     | [MiniCPM-o-4.5-zhenwu-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) | [MiniCPM-o-4.5-zhenwu-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) |

###### 综合评估

**Transformers–FlagOS 版本**

在多后端环境下使用 `USE_FLAGOS=1` 与在 NVIDIA CUDA 上使用 `USE_FLAGOS=0` 的精度差异

| 指标                       |        FlagOS 后端         | 与 Nvidia-CUDA 相比差异 |
|:-------------------------|:---------------:|:---------------------------:|
| Video-MME 0-shot avg@1 ↑ |     Nvidia      |            0.33%            |
| Video-MME 0-shot avg@1 ↑ |  Hygon-BW1000   |            0.17%            |
| Video-MME 0-shot avg@1 ↑ |    Ascend-A3    |            0.50%            |
| Video-MME 0-shot avg@1 ↑ | Iluvatar-BIV150 |            1.83%            |
| Video-MME 0-shot avg@1 ↑ |   Metax-C550    |            0.75%            |

**VLLM-FlagOS 版本**

在 NVIDIA 上使用 `USE_FLAGGEMS=1 FLAGCX_PATH=/workspace/FlagCX`，或在真武 810E `USE_FLAGGEMS=1`，与直接在 NVIDIA 平台上启动 vLLM Server 的精度差异

| 指标 (avg@1)          | Nvidia-FlagOS 与 Nvidia-CUDA 的差异 | zhenwu-FlagOS 与 Nvidia-CUDA 的差异 |
|:--------------------|:------------------------------------------------:|:------------------------------------------------:|
| CMMMU ↑             | 0.72% | 3.5% |
| MMMU ↑              | 1.44% | 1.18% |
| MMMU_Pro_standard ↑ | 0.83% | 0.22% |
| MM-Vet v2 ↑         | 0.46% | 1.33% |
| OCRBench ↑          | 0.10% | 1% |
| CII-Bench ↑         | 0.40% | 0.13% |
| Blink ↑             | 1.90% | 2.19% |


##### FlagOS 使用方式

###### 使用 FlagOS 在Nvidia体验性能加速

在Transformers版本上，CUDA生态与FlagOS生态精度对齐的前提下，FlagOS相比CUDA任务的负载执行总时间有6%的性能提升。

**From FlagRelease【推荐】**

FlagRelease是FlagOS团队构建的一套面向多架构人工智能芯片的大模型自动迁移、适配与发布平台，已发布MiniCPM-o-4.5的多芯片版本。FlagRelase已内置相关软件包，无需用户安装。

- FlagRelease 镜像关键版本信息

  | 组件                      | 版本                                |
  |:------------------------|:------------------------------------|
  | 加速卡驱动                | 570.158.01                          |
  | CUDA SDK Build          | cuda_13.0.r13.0/compiler.36424714_0 |
  | FlagTree                | 0.4.0+3.5                           |
  | FlagGems                | 4.2.1rc0                            |
  | vllm & vllm-plugin-fl   | 0.13.0 + vllm_fl 0.0.0              |
  | FlagCX                  | 0.1.0                               |  

- FlagRelease 使用速递

  | Vendor     | ModelScope   | Huggingface  |
  |:-----------|:------------:|:------------:|
  | Nvidia | [MiniCPM-o-4.5-nvidia-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) | [MiniCPM-o-4.5-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) |
  | Hygon-BW1000 | [MiniCPM-o-4.5-hygon-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) | [MiniCPM-o-4.5-hygon-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-hygon-FlagOS) |
  | Metax-C550 | [MiniCPM-o-4.5-metax-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) | [MiniCPM-o-4.5-metax-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-metax-FlagOS) |
  | Iluvatar-BIV150 | [MiniCPM-o-4.5-iluvatar-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) | [MiniCPM-o-4.5-iluvatar-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-iluvatar-FlagOS) |
  | Ascend-A3 | [MiniCPM-o-4.5-ascend-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) | [MiniCPM-o-4.5-ascend-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-ascend-FlagOS) |
  | Zhenwu-810E | [MiniCPM-o-4.5-zhenwu-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) | [MiniCPM-o-4.5-zhenwu-FlagOS](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-zhenwu-FlagOS) |  


###### 从零开始

- 依赖Python3.12, GLIBC_2.39, GLIBCXX_3.4.33, CXXABI_1.3.15 环境

**Transformers 版本**

- 安装FlagOS算子库

  官方仓库：https://github.com/flagos-ai/FlagGems

  ```shell
  pip install flag-gems==4.2.1rc0
  ```

- 安装FlagOS编译器

  官方仓库：https://github.com/flagos-ai/flagtree

  底层依赖库版本速查：https://github.com/flagos-ai/FlagTree/blob/main/documents/build.md#tips-for-building

  ```shell
  pip uninstall triton
  
  python3 -m pip install flagtree==0.4.0+3.5 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net
  ```

- 开启加速

  在欲执行任务的命令前添加 `USE_FLAGOS=1`，例如，当您通过
  ```shell
  python3 generate_speech_from_video.py
  ```
  
  来使用 MiniCPM-o-4.5 模型根据视频内容生成语音回答时，可以通过
  ```shell
  USE_FLAGOS=1 python3 generate_speech_from_video.py
  ```
  来使用 FlagOS 加速这一过程。

**Vllm 版本**

- 安装FlagOS算子库

  官方仓库：https://github.com/flagos-ai/FlagGems
  ```shell
  pip install flag-gems==4.2.1rc0
  pip install triton==3.5.1
  ```

- 开启加速
  在欲执行任务的命令前添加 `USE_FLAGOS=1`，例如，当您通过
  ```shell
  vllm serve ${model_path} --dtype auto  --gpu_memory_utilization 0.9 --trust-remote-code --max-num-batched-tokens 2048 --served-model-name cpmo --port ${Port}
  ```

  来启动MiniCPM-o-4.5服务端时，可以通过
  ```shell
  USE_FLAGOS=1 vllm serve ${model_path} --dtype auto  --gpu_memory_utilization 0.9 --trust-remote-code --max-num-batched-tokens 2048 --served-model-name cpmo --port ${Port}
  ```
  来使用FlagOS加速这一过程。

##### 使用 FlagOS 统一多芯片后端插件

[vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) 是一个为 vLLM 推理/服务框架构建的插件，它基于 FlagOS 的统一多芯片后端 开发，旨在扩展 vLLM 在多种硬件环境下的功能和性能表现。

###### Using vllm-plugin-FL

| 厂商   | 从零开始                                                                                                                 | 从 FlagRelease 开始                                                                                               |
|:-------|:-----------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
| Nvidia | [vllm-plugin-FL/MiniCPM-o-4.5](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/minicpm/README.md) | [MiniCPM-o-4.5-ModelScope](https://modelscope.cn/models/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS), [MiniCPM-o-4.5-Huggingface](https://huggingface.co/FlagRelease/MiniCPM-o-4.5-nvidia-FlagOS) |

</details>



### 训练：LLaMA-Factory、SWIFT

MiniCPM-V 和 MiniCPM-o 模型支持通过 LLaMA-Factory 和 SWIFT 等框架训练。具体用法可以参考我们的[使用指南](https://github.com/OpenSQZ/MiniCPM-V-Cookbook)。

| 框架 | MiniCPM-V 4.7 | 更多 MiniCPM-V/o 模型 |
|:---|:---:|:---:|
| LLaMA-Factory | [微调指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/tree/main/finetune/llamafactory) | [微调指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/tree/main/finetune/llamafactory) |
| SWIFT | [微调指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/swift/swift_minicpmv47_35b_zh.md) | [微调指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/tree/main/finetune/swift) |


### API

MiniCPM-V 4.7 / 4.5 和 MiniCPM-o 4.5 的 API 用法见 [API Guide](./api.md)。MiniCPM-V 4.6 的 API 入口已保存在[独立文档](./minicpm_v4dot6_zh.md#api)中。

| 模型 | API |
|:---|:---:|
| MiniCPM-V 4.7 | [指南](./api.md#minicpm-v-47) |
| MiniCPM-V 4.5 | [指南](./api.md#minicpm-v-45) |
| MiniCPM-o 4.5 | [指南](./api.md#minicpm-o-45) |
