## Phi-3-vision-128K-Instruct vs MiniCPM-Llama3-V 2.5

Comparison results of Phi-3-vision-128K-Instruct and MiniCPM-Llama3-V 2.5, regarding the model size, hardware requirements, and performances on multiple popular benchmarks.

我们提供了从模型参数、硬件需求、全面性能指标等方面对比 Phi-3-vision-128K-Instruct 和 MiniCPM-Llama3-V 2.5 的结果。
 
 ## Hardeware Requirements （硬件需求）

With in4 quantization, MiniCPM-Llama3-V 2.5 delivers smooth inference of 6-8 tokens/s on edge devices with only 8GB of GPU memory.

通过 in4 量化，MiniCPM-Llama3-V 2.5 仅需 8GB 显存即可提供端侧 6-8 tokens/s 的流畅推理。

| Model（模型）                | GPU Memory（显存）        |
|:----------------------|:-------------------:|
| [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/)  | 19 GB   |
| Phi-3-vision-128K-Instruct | 12 GB |
| [MiniCPM-Llama3-V 2.5 (int4)](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4/)  | 8 GB |

## Model Size and Peformance （模型参数和性能）



| | Phi-3-vision-128K-Instruct | MiniCPM-Llama3-V 2.5|
|:-|:----------:|:-------------------:|
| Size（参数） | **4B** | 8B|
| OpenCompass 2024/05 | 53.7 | **58.8** |
| OCRBench | 639.0  | **725.0**|
| RealworldQA | 58.8 | **63.5**|
| TextVQA | 72.2 | **76.6** |
| ScienceQA| **90.8** | 89.0 | 
| POPE | 83.4 | **87.2** |