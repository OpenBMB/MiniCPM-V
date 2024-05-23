## Phi-3-vision-128K-Instruct vs MiniCPM-Llama3-V 2.5

Comparison results of Phi-3-vision-128K-Instruct and MiniCPM-Llama3-V 2.5, regarding the model size, hardware requirements, and performances.

我们提供了从模型参数、硬件需求、性能指标等方面对比 Phi-3-vision-128K-Instruct 和 MiniCPM-Llama3-V 2.5 的结果。
 
 ## Hardeware Requirements （硬件需求）

With int4 quantization, MiniCPM-Llama3-V 2.5 delivers smooth inference with only 8GB of GPU memory.

通过 int4 量化，MiniCPM-Llama3-V 2.5 仅需 8GB 显存即可推理。

| Model（模型）                | GPU Memory（显存）        |
|:----------------------|:-------------------:|
| [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/)  | 19 GB   |
| Phi-3-vision-128K-Instruct | 12 GB |
| [MiniCPM-Llama3-V 2.5 (int4)](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4/)  | 8 GB |

## Model Size and Peformance （模型参数和性能）

In most benchmarks, MiniCPM-Llama3-V 2.5 achieves **better performance** compared with Phi-3-vision-128K-Instruct.

在大多数评测集上， MiniCPM-Llama3-V 2.5 相比于 Phi-3-vision-128K-Instruct 都展现出了**更优的性能表现**.

| | Phi-3-vision-128K-Instruct | MiniCPM-Llama3-V 2.5|
|:-|:----------:|:-------------------:|
| Size（参数） | **4B** | 8B|
| First Token Latency（首token延迟）$^1$ | L: 330ms, M: 330ms, H: 330ms | **L: 48ms, M: 145ms, H: 278ms** |
| Throughtput（吞吐率）$^2$| 30 tokens/s | **41 tokens/s**|
| OpenCompass 2024/05 | 53.7 | **58.8** |
| OCRBench | 639.0  | **725.0**|
| RealworldQA | 58.8 | **63.5**|
| TextVQA | 72.2 | **76.6** |
| ScienceQA| **90.8** | 89.0 | 
| POPE | 83.4 | **87.2** |
| MathVista | 44.5 | **54.3** |
| MMStar | 47.4 | **51.8** |
| LLaVA Bench | 64.2 | **86.7** |
| AI2D | 76.7 | **78.4** |

<small>
1: L(ow): 448pxl, M(edium): 896pxl, H(igh): 1344pxl input images.
<br>
2. Evaluation environment: A800 GPU, flash-attn=2.4.3, batch-size=1.
</small>


## Multilingual Capabilities


MiniCPM-Llama3-V 2.5 exhibits **stronger multilingual** capabilities compared with Phi-3-vision-128K-Instruct on LLaVA Bench.

MiniCPM-Llama3-V 2.5 在对话和推理评测榜单 LLaVA Bench 上展现出了比 Phi-3-vision-128K-Instruct **更强的多语言的性能**。

<div align="center">
    <img src="../assets/llavabench_compare_phi-3.png" width="85%" />
    <br>
    Evaluation results of LLaVABench in multiple languages
    <br>
    多语言LLaVA Bench评测结果
</div>
