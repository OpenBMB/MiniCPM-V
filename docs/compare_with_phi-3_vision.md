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

In most benchmarks, MiniCPM-Llama3-V 2.5 achieves **better performance** compared with Phi-3-vision-128K-Instruct. Moreover, MiniCPM-Llama3-V 2.5 also exhibits **lower latency and better throughtput even without quantization**.

在大多数评测集上， MiniCPM-Llama3-V 2.5 相比于 Phi-3-vision-128K-Instruct 都展现出了**更优的性能表现**。 即使未经量化，MiniCPM-Llama3-V 2.5 的**推理延迟和吞吐率也都更具优势**。

| | Phi-3-vision-128K-Instruct | MiniCPM-Llama3-V 2.5|
|:-|:----------:|:-------------------:|
| Size（参数） | **4B** | 8B|
| First Token Latency（首token延迟）<sup>2</sup> | L: 330ms, M: 330ms, H: 330ms | **L: 48ms, M: 145ms, H: 278ms** |
| Throughtput（吞吐率）<sup>2</sup>| 30 tokens/s | **41 tokens/s**|
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
1. Evaluation environment: A800 GPU, flash-attn=2.4.3, batch-size=1.
<br>
<span style="color:grey">
MiniCPM-Llama3-V 2.5 shows better first token latency and throughput performance even though the number of parameters is twice as large as that of Phi-3-vision-128k-instruct due to its efficient image encoding method and adaptive resolution encoding strategy.  For example, for an input images with a 448x448 resolution, MiniCPM-Llama3-V 2.5 encodes it into 96 tokens, while Phi-3-vision-128k-instruct encodes it into 2500+ tokens. Longer image token sequence significantly affects the first token latency and throughput. MiniCPM-V series models insist on obtaining stronger performance with more efficient encoding, thus achieves efficient end-device deployment and providing better experience for end users.
<br>
得益于 MiniCPM-Llama3-V 2.5 高效的图像编码方式和自适应分辨率编码策略，即使参数量比 Phi-3-vision-128k-instruct 大一倍，依然展现出了更出色的首 token 延迟和吞吐量表现。例如两个模型对输入分辨率为448x448 的图像，MiniCPM-Llama3-V 2.5 的图像编码长度为 96， 而Phi-3-vision-128k-instruct 的图像编码长度为 2500+。更长的图像编码长度会显著影响首token延迟和吞吐量，MiniCPM-V系列坚持用更高效的编码方式撬动更强的性能，进而实现高效的终端设备部署，为端侧用户提供更良好的体验。
</span>
</small>


## Multilingual Capabilities


MiniCPM-Llama3-V 2.5 exhibits **stronger multilingual** capabilities compared with Phi-3-vision-128K-Instruct on LLaVA Bench.

MiniCPM-Llama3-V 2.5 在对话和推理评测榜单 LLaVA Bench 上展现出了比 Phi-3-vision-128K-Instruct **更强的多语言的性能**。

<div align="center">
    <img src="../assets/llavabench_compare_phi3.png" width="85%" />
    <br>
    Evaluation results of LLaVABench in multiple languages
    <br>
    多语言LLaVA Bench评测结果
</div>
