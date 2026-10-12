[← 首页](../README_zh.md) · [English](./minicpm_o4dot5_en.md)

# MiniCPM-o 4.5

[性能评估](#性能评估) · [典型示例](#典型示例) · [Transformers 推理](#基于-transformers-推理) · [本地 Demo](#本地-demo-部署) · [模型权重](#模型权重) · [模型局限](#模型局限性)

除示例另有说明外，以下代码均从仓库根目录运行。

**MiniCPM-o 4.5** 是 MiniCPM-o 系列中最新且性能最强的模型。该模型采用端到端方式构建，基于 SigLip2、Whisper-medium、CosyVoice2 和 Qwen3-8B，总参数量为 9B。其在性能上实现了显著提升，并引入了全新的全双工多模态实时流式交互能力。MiniCPM-o 4.5 的主要特性包括：

- 🔥 **领先的视觉能力**
  MiniCPM-o 4.5 在涵盖 8 个主流评测基准的 OpenCompass 综合评估中获得了 77.6 的平均分。**仅凭 9B 参数，其视觉理解能力超越了 GPT-4o、Gemini 2.0 Pro 等广泛使用的商业模型**，接近 Gemini 2.5 Flash 水平。 该模型在单一模型中同时支持指令模式和思考模式，能够更好地平衡不同使用场景下的效率与性能。
- 🎙 **强大的语音能力**
  MiniCPM-o 4.5 支持**音色可配置的中英双语实时语音对话**。其语音对话**更加自然、富有表现力且稳定**。该模型还支持更多有趣的功能，如**通过简单的参考音频进行声音克隆和角色扮演**，其声音克隆表现甚至超越了 CosyVoice2 等优秀 TTS 工具。
- 🎬 **全双工及主动多模态实时流式交互能力**
  MiniCPM-o 4.5 的一项核心新特性是能够以端到端的方式同时处理实时连续的视频和音频输入流，并同步生成文本和语音输出流，且互不阻塞。这**使得 MiniCPM-o 4.5 能够同时“看、听、说”**，提供流畅的实时全模态对话体验。除了被动响应，模型还能进行**主动交互**，例如基于对场景的实时持续理解，主动发起提醒或评论。
- 💪 **高效率、强悍的 OCR 能力及其他特性**
  MiniCPM-o 4.5 进一步优化了 MiniCPM-V 系列的众多视觉能力，可以高效处理**任意长宽比的高分辨率图像**（最高 180 万像素）和**高帧率视频**（最高 10fps）。其在 OmniDocBench **端到端英文文档解析测试中达到了业内顶尖水平**，超越了 Gemini-3 Flash 和 GPT-5 等商业模型以及 DeepSeek-OCR 2 等专用工具。此外，它还具备**可信的多模态行为**，在 MMHal-Bench 上与 Gemini 2.5 Flash 相当，并**支持超过 30 种语言**。
-  💫  **便捷的使用体验**
  MiniCPM-o 4.5 提供了多种便捷的使用方式：**基础用法（推荐，100% 精度）：** 基于 PyTorch 的 Nvidia GPU 推理。**其他端侧适配**包括：(1) 支持 llama.cpp 和 Ollama，以便在本地设备上进行高效的 CPU 推理；(2) 提供 16 种尺寸的 int4 和 GGUF 格式量化模型；(3) 支持 vLLM 和 SGLang，实现高吞吐、显存高效的推理；(4) 支持 FlagOS 统一多芯片后端插件。**我们还开源了 Web Demo**，**让全双工多模态实时流式交互体验在 GPU、PC（如 MacBook）等本地设备上触手可及**。

**模型架构。**

- **端到端全模态架构。** 各模态的编码器/解码器与大语言模型通过稠密特征以端到端的方式进行紧密连接。这种设计实现了更好的信息流转与控制，有助于在训练过程中充分挖掘和利用丰富的多模态知识。
- **全双工多模态实时流机制。** （1）我们将离线模态编码器/解码器转化为支持流式输入/输出的在线全双工版本。语音解码器采用文本与语音 token 交错建模的方式，支持全双工语音生成（即与新输入实时同步），同时也提升了长语音（如超过 1 分钟）生成的稳定性。（2）时分复用：**我们在毫秒级时间线上同步所有输入和输出流**，并利用时分复用机制在语言模型主干中进行统一建模。该机制将并行的全模态流划分为微小周期性时间片内的顺序信息组，从而实现高效的全模态流式处理。
- **主动交互机制。** 语言模型模块会持续监控输入的视频和音频流，并以 1Hz 的频率自动决策是否发言。这种高频决策能力结合全双工特性，是实现主动提醒、主动评论等“主动交互”能力的关键。
- **可配置语音建模设计。** 我们延续了 MiniCPM-o 2.6 的多模态系统提示词设计，同时包含文本系统提示词和音频系统提示词（用于指定音色）。这使得模型在推理阶段能够通过简单的参考音频实现声音克隆和角色扮演。

<div align="center">
  <img src="../assets/minicpm-o-45-framework.png", width=100%>
</div>

### 性能评估


<div align="center">
  <img src="../assets/radar_minicpmo4.5.png", width=80%>
</div>

&emsp;
<br>

<details>
<summary>点击查看 MiniCPM-o 4.5 详细评测拆解。</summary>


<div align="center">
  <img src="../assets/minicpm_o_45_main_exp_table.png", width=90%>
</div>
<strong>说明</strong>: * 为自测结果，其余为引用的公开结果。n/a 表示该模型不支持对应模态或任务。所有结果来自指令模式或对应模型指令版本权重。


<details>
<summary>点击查看视觉理解能力详细评测结果。</summary>

**图像理解能力（指令模式）**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>OpenCompass</b></th>
  <th nowrap="nowrap"><b>MMBench EN v1.1</b></th>
  <th nowrap="nowrap"><b>MMBench CN v1.1</b></th>
  <th nowrap="nowrap"><b>MathVista</b></th>
  <th nowrap="nowrap"><b>MMVet</b></th>
  <th nowrap="nowrap"><b>MMMU</b></th>
  <th nowrap="nowrap"><b>MMStar</b></th>
  <th nowrap="nowrap"><b>HallusionBench</b></th>
  <th nowrap="nowrap"><b>AI2D</b></th>
  <th nowrap="nowrap"><b>OCRBench</b></th>
  <th nowrap="nowrap"><b>TextVQA_VAL</b></th>
  <th nowrap="nowrap"><b>DocVQA_VAL</b></th>
  <th nowrap="nowrap"><b>MMT-Bench_VAL</b></th>
  <th nowrap="nowrap"><b>MM-IFEval</b></th>
  <th nowrap="nowrap"><b>Mantis-Eval</b></th>
  <th nowrap="nowrap"><b>MuirBench</b></th>
  <th nowrap="nowrap"><b>MMSI-Bench</b></th>
  <th nowrap="nowrap"><b>MMHal-Score</b></th>
  <th nowrap="nowrap"><b>MMHal-Hallrate↓</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Gemini2.5-Flash-Nonthinking</td>
  <td align="center"><b>78.5</b></td>
  <td align="center"><ins>86.6</ins></td>
  <td align="center"><ins>86.0</ins></td>
  <td align="center">75.3</td>
  <td align="center"><ins>81.4</ins><sup>*</sup></td>
  <td align="center"><b>76.3</b></td>
  <td align="center"><b>75.8</b></td>
  <td align="center">59.1</td>
  <td align="center"><b>87.7</b></td>
  <td align="center">864</td>
  <td align="center">74.3<sup>*</sup></td>
  <td align="center">93.0</td>
  <td align="center"><ins>70.0</ins><sup>*</sup></td>
  <td align="center"><b>75.8<sup>*</sup></b></td>
  <td align="center">72.8<sup>*</sup></td>
  <td align="center"><b>74.5<sup>*</sup></b></td>
  <td align="center">12.1<sup>*</sup></td>
  <td align="center"><ins>4.6</ins><sup>*</sup></td>
  <td align="center"><b>23.9<sup>*</sup></b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Gemini2.0-Pro</td>
  <td align="center">73.3</td>
  <td align="center">83.0</td>
  <td align="center">83.0</td>
  <td align="center">71.3</td>
  <td align="center">70.4</td>
  <td align="center">72.6</td>
  <td align="center">68.5</td>
  <td align="center">49.8</td>
  <td align="center">84.8</td>
  <td align="center">863</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">GPT-4o</td>
  <td align="center">75.4</td>
  <td align="center">86.0</td>
  <td align="center"><ins>86.0</ins></td>
  <td align="center">71.6</td>
  <td align="center">76.9</td>
  <td align="center">72.9</td>
  <td align="center">70.2</td>
  <td align="center">57.0</td>
  <td align="center">86.3</td>
  <td align="center">822</td>
  <td align="center">77.4</td>
  <td align="center">93.0</td>
  <td align="center">66.7<sup>*</sup></td>
  <td align="center">64.6</td>
  <td align="center">70.1<sup>*</sup></td>
  <td align="center">70.5<sup>*</sup></td>
  <td align="center">8.1<sup>*</sup></td>
  <td align="center">4.2<sup>*</sup></td>
  <td align="center">25.0<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">InternVL-3.5-8B</td>
  <td align="center">75.8</td>
  <td align="center">79.5</td>
  <td align="center">80.0<sup>*</sup></td>
  <td align="center"><ins>78.4</ins></td>
  <td align="center"><b>83.1</b></td>
  <td align="center"><ins>73.4</ins></td>
  <td align="center">69.3</td>
  <td align="center">54.5</td>
  <td align="center">84.0</td>
  <td align="center">840</td>
  <td align="center">78.2</td>
  <td align="center">92.3</td>
  <td align="center">66.7</td>
  <td align="center">56.3<sup>*</sup></td>
  <td align="center">70.5</td>
  <td align="center">55.8</td>
  <td align="center">-</td>
  <td align="center">3.8<sup>*</sup></td>
  <td align="center">34.7<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-VL-8B-Instruct</td>
  <td align="center">76.5</td>
  <td align="center">84.5</td>
  <td align="center">84.7</td>
  <td align="center">77.2</td>
  <td align="center">73.7<sup>*</sup></td>
  <td align="center">69.6</td>
  <td align="center">70.9</td>
  <td align="center"><ins>61.1</ins></td>
  <td align="center">85.7</td>
  <td align="center"><b>896</b></td>
  <td align="center">82.9<sup>*</sup></td>
  <td align="center"><b>96.1</b></td>
  <td align="center">60.9<sup>*</sup></td>
  <td align="center">59.4<sup>*</sup></td>
  <td align="center">74.2<sup>*</sup></td>
  <td align="center">64.4</td>
  <td align="center">11.3<sup>*</sup></td>
  <td align="center"><b>4.7<sup>*</sup></b></td>
  <td align="center">29.9<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center">75.7</td>
  <td align="center">84.9<sup>*</sup></td>
  <td align="center">84.1<sup>*</sup></td>
  <td align="center">75.9</td>
  <td align="center">74.8<sup>*</sup></td>
  <td align="center">69.1</td>
  <td align="center">68.5</td>
  <td align="center">59.7</td>
  <td align="center">85.2</td>
  <td align="center"><ins>880</ins><sup>*</sup></td>
  <td align="center"><b>84.1<sup>*</sup></b></td>
  <td align="center"><ins>95.4</ins><sup>*</sup></td>
  <td align="center"><b>70.4<sup>*</sup></b></td>
  <td align="center">65.7<sup>*</sup></td>
  <td align="center"><ins>78.3</ins><sup>*</sup></td>
  <td align="center">61.9<sup>*</sup></td>
  <td align="center"><ins>14.2</ins><sup>*</sup></td>
  <td align="center"><ins>4.6</ins><sup>*</sup></td>
  <td align="center">31.6<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><ins>77.6</ins></td>
  <td align="center"><b>87.6</b></td>
  <td align="center"><b>87.2</b></td>
  <td align="center"><b>80.1</b></td>
  <td align="center">74.4</td>
  <td align="center">67.6</td>
  <td align="center"><ins>73.1</ins></td>
  <td align="center"><b>63.2</b></td>
  <td align="center"><ins>87.6</ins></td>
  <td align="center">876</td>
  <td align="center"><ins>83.8</ins></td>
  <td align="center">94.7</td>
  <td align="center">69.7</td>
  <td align="center"><ins>66.3</ins></td>
  <td align="center"><b>79.7</b></td>
  <td align="center"><ins>72.0</ins></td>
  <td align="center"><b>16.6</b></td>
  <td align="center"><b>4.7</b></td>
  <td align="center"><ins>24.3</ins></td>
</tr>
  </table>
  </div>
  
**图像理解能力（思考模式）**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>OpenCompass</b></th>
  <th nowrap="nowrap"><b>MMBench EN v1.1</b></th>
  <th nowrap="nowrap"><b>MMBench CN v1.1</b></th>
  <th nowrap="nowrap"><b>MathVista</b></th>
  <th nowrap="nowrap"><b>MMVet</b></th>
  <th nowrap="nowrap"><b>MMMU</b></th>
  <th nowrap="nowrap"><b>MMStar</b></th>
  <th nowrap="nowrap"><b>HallusionBench</b></th>
  <th nowrap="nowrap"><b>AI2D</b></th>
  <th nowrap="nowrap"><b>OCRBench</b></th>
  <th nowrap="nowrap"><b>TextVQA_VAL</b></th>
  <th nowrap="nowrap"><b>DocVQA_VAL</b></th>
  <th nowrap="nowrap"><b>MMT-Bench_VAL</b></th>
  <th nowrap="nowrap"><b>MM-IFEval</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Gemini2.5-Flash-Thinking</td>
  <td align="center"><b>79.9</b></td>
  <td align="center">87.1</td>
  <td align="center">87.3</td>
  <td align="center">79.4</td>
  <td align="center"><b>81.2<sup>*</sup></b></td>
  <td align="center"><ins>77.7</ins></td>
  <td align="center"><b>76.5</b></td>
  <td align="center">63.5</td>
  <td align="center"><ins>88.7</ins></td>
  <td align="center">853</td>
  <td align="center">73.8<sup>*</sup></td>
  <td align="center">92.8</td>
  <td align="center">70.7<sup>*</sup></td>
  <td align="center"><ins>75.7</ins><sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">GPT-5</td>
  <td align="center"><ins>79.7</ins></td>
  <td align="center">85.5<sup>*</sup></td>
  <td align="center">85.6<sup>*</sup></td>
  <td align="center"><b>81.9</b></td>
  <td align="center"><ins>77.6</ins></td>
  <td align="center"><b>81.8</b></td>
  <td align="center"><ins>75.7</ins></td>
  <td align="center"><ins>65.2</ins></td>
  <td align="center"><b>89.5</b></td>
  <td align="center">807</td>
  <td align="center">77.8<sup>*</sup></td>
  <td align="center">91.3<sup>*</sup></td>
  <td align="center"><b>72.7<sup>*</sup></b></td>
  <td align="center"><b>83.1<sup>*</sup></b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-VL-8B-Thinking</td>
  <td align="center">77.3</td>
  <td align="center">85.3</td>
  <td align="center">85.5</td>
  <td align="center"><ins>81.4</ins></td>
  <td align="center">69.8<sup>*</sup></td>
  <td align="center">74.1</td>
  <td align="center">75.3</td>
  <td align="center"><b>65.4</b></td>
  <td align="center">84.9</td>
  <td align="center">819</td>
  <td align="center">77.8<sup>*</sup></td>
  <td align="center"><b>95.3</b></td>
  <td align="center">68.1<sup>*</sup></td>
  <td align="center">73.5<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Thinking</td>
  <td align="center">78.5</td>
  <td align="center"><ins>88.2</ins><sup>*</sup></td>
  <td align="center"><b>87.7<sup>*</sup></b></td>
  <td align="center">80.0</td>
  <td align="center">74.8<sup>*</sup></td>
  <td align="center">75.6</td>
  <td align="center">74.9</td>
  <td align="center">62.8</td>
  <td align="center">86.1</td>
  <td align="center"><ins>859</ins><sup>*</sup></td>
  <td align="center"><b>80.8<sup>*</sup></b></td>
  <td align="center"><ins>94.2</ins><sup>*</sup></td>
  <td align="center"><ins>70.9</ins><sup>*</sup></td>
  <td align="center">69.9<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Thinking</td>
  <td align="center">78.2</td>
  <td align="center"><b>89.0</b></td>
  <td align="center"><ins>87.6</ins></td>
  <td align="center">81.0</td>
  <td align="center">73.6</td>
  <td align="center">70.2</td>
  <td align="center">73.6</td>
  <td align="center">62.6</td>
  <td align="center">88.5</td>
  <td align="center"><b>879</b></td>
  <td align="center"><ins>79.8</ins></td>
  <td align="center">92.3</td>
  <td align="center">69.7</td>
  <td align="center">68.2</td>
</tr>
  </table>
  </div>

**视频理解能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>Video-MME<br>(w/o subs)</b></th>
  <th nowrap="nowrap"><b>LVBench</b></th>
  <th nowrap="nowrap"><b>MLVU<br>(M-Avg)</b></th>
  <th nowrap="nowrap"><b>LongVideoBench<br>(val)</b></th>
  <th nowrap="nowrap"><b>MotionBench</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Gemini2.5-Flash-Nonthinking</td>
  <td align="center"><b>75.6</b></td>
  <td align="center"><b>62.2</b></td>
  <td align="center"><b>77.8</b></td>
  <td align="center">-</td>
  <td align="center">-</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">InternVL-3.5-8B</td>
  <td align="center">66.0</td>
  <td align="center">-</td>
  <td align="center">70.2</td>
  <td align="center">62.1</td>
  <td align="center"><b>62.3<sup>*</sup></b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center"><ins>70.5</ins></td>
  <td align="center">50.2</td>
  <td align="center">75.2</td>
  <td align="center"><b>66.9<sup>*</sup></b></td>
  <td align="center"><ins>61.7</ins><sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center">70.4</td>
  <td align="center"><ins>50.9</ins></td>
  <td align="center"><ins>76.5</ins></td>
  <td align="center"><ins>66.0</ins></td>
  <td align="center">61.4</td>
</tr>
  </table>
  </div>

</details>

<details>
<summary>点击查看文档解析能力详细评测结果。</summary>

**OmniDocBench**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left" rowspan="2"><b>Method Type</b></th>
  <th nowrap="nowrap" rowspan="2"><b>Methods</b></th>
  <th nowrap="nowrap" colspan="2"><b>OverallEdit↓</b></th>
  <th nowrap="nowrap" colspan="2"><b>TextEdit↓</b></th>
  <th nowrap="nowrap" colspan="2"><b>FormulaEdit↓</b></th>
  <th nowrap="nowrap" colspan="2"><b>TableTEDS↑</b></th>
  <th nowrap="nowrap" colspan="2"><b>TableEdit↓</b></th>
  <th nowrap="nowrap" colspan="2"><b>Read OrderEdit↓</b></th>
</tr>
<tr>
  <th nowrap="nowrap"><b>EN</b></th>
  <th nowrap="nowrap"><b>ZH</b></th>
  <th nowrap="nowrap"><b>EN</b></th>
  <th nowrap="nowrap"><b>ZH</b></th>
  <th nowrap="nowrap"><b>EN</b></th>
  <th nowrap="nowrap"><b>ZH</b></th>
  <th nowrap="nowrap"><b>EN</b></th>
  <th nowrap="nowrap"><b>ZH</b></th>
  <th nowrap="nowrap"><b>EN</b></th>
  <th nowrap="nowrap"><b>ZH</b></th>
  <th nowrap="nowrap"><b>EN</b></th>
  <th nowrap="nowrap"><b>ZH</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left" rowspan="2">Pipeline</td>
  <td align="center">MinerU 2.5</td>
  <td align="center">0.117<sup>*</sup></td>
  <td align="center">0.172<sup>*</sup></td>
  <td align="center">0.051<sup>*</sup></td>
  <td align="center">0.08<sup>*</sup></td>
  <td align="center"><ins>0.256</ins><sup>*</sup></td>
  <td align="center">0.455<sup>*</sup></td>
  <td align="center">85.9<sup>*</sup></td>
  <td align="center">89.4<sup>*</sup></td>
  <td align="center">0.115<sup>*</sup></td>
  <td align="center">0.081<sup>*</sup></td>
  <td align="center">0.047<sup>*</sup></td>
  <td align="center">0.072<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">PaddleOCR-VL</td>
  <td align="center"><b>0.105</b></td>
  <td align="center"><ins>0.126</ins></td>
  <td align="center"><ins>0.041</ins></td>
  <td align="center"><b>0.062</b></td>
  <td align="center"><b>0.241</b></td>
  <td align="center"><b>0.316</b></td>
  <td align="center">88</td>
  <td align="center"><ins>92.1</ins></td>
  <td align="center"><ins>0.093</ins></td>
  <td align="center"><ins>0.062</ins></td>
  <td align="center">0.045</td>
  <td align="center"><ins>0.063</ins></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
  <td align="center"></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left" rowspan="11">End-to-end Model</td>
  <td align="center">Qwen2.5-VL-72B</td>
  <td align="center">0.214</td>
  <td align="center">0.261</td>
  <td align="center">0.092</td>
  <td align="center">0.18</td>
  <td align="center">0.315</td>
  <td align="center">0.434</td>
  <td align="center">82.9</td>
  <td align="center">83.9</td>
  <td align="center">0.341</td>
  <td align="center">0.262</td>
  <td align="center">0.106</td>
  <td align="center">0.168</td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">GPT 5</td>
  <td align="center">0.218<sup>*</sup></td>
  <td align="center">0.33<sup>*</sup></td>
  <td align="center">0.139<sup>*</sup></td>
  <td align="center">0.344<sup>*</sup></td>
  <td align="center">0.396<sup>*</sup></td>
  <td align="center">0.555<sup>*</sup></td>
  <td align="center">77.55<sup>*</sup></td>
  <td align="center">73.09<sup>*</sup></td>
  <td align="center">0.188<sup>*</sup></td>
  <td align="center">0.196<sup>*</sup></td>
  <td align="center">0.151<sup>*</sup></td>
  <td align="center">0.227<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">Gemini2.5-Flash-Nonthinking</td>
  <td align="center">0.214<sup>*</sup></td>
  <td align="center">0.29<sup>*</sup></td>
  <td align="center">0.159<sup>*</sup></td>
  <td align="center">0.273<sup>*</sup></td>
  <td align="center">0.368<sup>*</sup></td>
  <td align="center">0.524<sup>*</sup></td>
  <td align="center">80.9<sup>*</sup></td>
  <td align="center">85.5<sup>*</sup></td>
  <td align="center">0.197<sup>*</sup></td>
  <td align="center">0.167<sup>*</sup></td>
  <td align="center">0.132<sup>*</sup></td>
  <td align="center">0.195<sup>*</sup></td>
</tr>
<tr>
  <td align="center">Gemini-2.5-Pro-Nonthinking</td>
  <td align="center">0.148<sup>*</sup></td>
  <td align="center">0.212<sup>*</sup></td>
  <td align="center">0.055<sup>*</sup></td>
  <td align="center">0.168<sup>*</sup></td>
  <td align="center">0.356<sup>*</sup></td>
  <td align="center">0.439<sup>*</sup></td>
  <td align="center">85.8<sup>*</sup></td>
  <td align="center">86.4<sup>*</sup></td>
  <td align="center">0.13<sup>*</sup></td>
  <td align="center">0.119<sup>*</sup></td>
  <td align="center">0.049<sup>*</sup></td>
  <td align="center">0.121<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">Gemini-3 Flash-Nonthinking</td>
  <td align="center">0.155<sup>*</sup></td>
  <td align="center">0.201<sup>*</sup></td>
  <td align="center">0.138<sup>*</sup></td>
  <td align="center">0.255<sup>*</sup></td>
  <td align="center">0.297<sup>*</sup></td>
  <td align="center">0.351<sup>*</sup></td>
  <td align="center">86.4<sup>*</sup></td>
  <td align="center">89.8<sup>*</sup></td>
  <td align="center">0.116<sup>*</sup></td>
  <td align="center">0.1<sup>*</sup></td>
  <td align="center">0.072<sup>*</sup></td>
  <td align="center">0.099<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">doubao-1-5-thinking-vision-pro-250428</td>
  <td align="center">0.14</td>
  <td align="center">0.162</td>
  <td align="center">0.043</td>
  <td align="center">0.085</td>
  <td align="center">0.295</td>
  <td align="center">0.384</td>
  <td align="center">83.3</td>
  <td align="center">89.3</td>
  <td align="center">0.165</td>
  <td align="center">0.085</td>
  <td align="center">0.058</td>
  <td align="center">0.094</td>
</tr>
<tr>
  <td align="center">dots.ocr</td>
  <td align="center">0.125</td>
  <td align="center">0.16</td>
  <td align="center"><b>0.032</b></td>
  <td align="center"><ins>0.066</ins></td>
  <td align="center">0.329</td>
  <td align="center">0.416</td>
  <td align="center"><ins>88.6</ins></td>
  <td align="center">89</td>
  <td align="center">0.099</td>
  <td align="center">0.092</td>
  <td align="center"><ins>0.04</ins></td>
  <td align="center">0.067</td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">HunyuanOCR</td>
  <td align="center">0.12<sup>*</sup></td>
  <td align="center"><b>0.125<sup>*</sup></b></td>
  <td align="center">0.046<sup>*</sup></td>
  <td align="center">0.071<sup>*</sup></td>
  <td align="center">0.288<sup>*</sup></td>
  <td align="center"><ins>0.33</ins><sup>*</sup></td>
  <td align="center"><b>89.6<sup>*</sup></b></td>
  <td align="center"><b>94.4<sup>*</sup></b></td>
  <td align="center"><b>0.089<sup>*</sup></b></td>
  <td align="center"><b>0.045<sup>*</sup></b></td>
  <td align="center">0.055<sup>*</sup></td>
  <td align="center"><b>0.056<sup>*</sup></b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">DeepSeek-OCR 2</td>
  <td align="center">0.119<sup>*</sup></td>
  <td align="center">0.146<sup>*</sup></td>
  <td align="center"><ins>0.041</ins><sup>*</sup></td>
  <td align="center">0.08<sup>*</sup></td>
  <td align="center"><ins>0.256</ins><sup>*</sup></td>
  <td align="center">0.345<sup>*</sup></td>
  <td align="center">82.6<sup>*</sup></td>
  <td align="center">89.9<sup>*</sup></td>
  <td align="center">0.123<sup>*</sup></td>
  <td align="center">0.078<sup>*</sup></td>
  <td align="center">0.055<sup>*</sup></td>
  <td align="center">0.081<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center">0.216<sup>*</sup></td>
  <td align="center">0.363<sup>*</sup></td>
  <td align="center">0.128<sup>*</sup></td>
  <td align="center">0.337<sup>*</sup></td>
  <td align="center">0.402<sup>*</sup></td>
  <td align="center">0.529<sup>*</sup></td>
  <td align="center">77.3<sup>*</sup></td>
  <td align="center">71.8<sup>*</sup></td>
  <td align="center">0.181<sup>*</sup></td>
  <td align="center">0.255<sup>*</sup></td>
  <td align="center">0.152<sup>*</sup></td>
  <td align="center">0.332<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="center">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><ins>0.109</ins></td>
  <td align="center">0.162</td>
  <td align="center">0.046</td>
  <td align="center">0.078</td>
  <td align="center">0.257</td>
  <td align="center">0.41</td>
  <td align="center">86.8</td>
  <td align="center">88.9</td>
  <td align="center">0.097</td>
  <td align="center">0.084</td>
  <td align="center"><b>0.037</b></td>
  <td align="center">0.074</td>
</tr>
  </table>
  </div>
</details>

<details>
<summary>点击查看文本能力详细评测结果。</summary>

**文本能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>IFEval-PLS</b></th>
  <th nowrap="nowrap"><b>BBH</b></th>
  <th nowrap="nowrap"><b>CMMLU</b></th>
  <th nowrap="nowrap"><b>MMLU</b></th>
  <th nowrap="nowrap"><b>HumanEval</b></th>
  <th nowrap="nowrap"><b>MBPP</b></th>
  <th nowrap="nowrap"><b>Math500</b></th>
  <th nowrap="nowrap"><b>GSM8K</b></th>
  <th nowrap="nowrap"><b>Avg</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-8B-Instruct</td>
  <td align="center">83.0<sup>*</sup></td>
  <td align="center">69.4<sup>*</sup></td>
  <td align="center">78.7<sup>*</sup></td>
  <td align="center"><b>81.7<sup>*</sup></b></td>
  <td align="center"><b>86.6<sup>*</sup></b></td>
  <td align="center">75.9<sup>*</sup></td>
  <td align="center"><b>84.0<sup>*</sup></b></td>
  <td align="center">93.4<sup>*</sup></td>
  <td align="center">81.6</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><b>84.7</b></td>
  <td align="center"><b>81.1</b></td>
  <td align="center"><b>79.5</b></td>
  <td align="center">77.0</td>
  <td align="center"><b>86.6</b></td>
  <td align="center"><b>76.7</b></td>
  <td align="center">77.0</td>
  <td align="center"><b>94.5</b></td>
  <td align="center"><b>82.1</b></td>
</tr>
  </table>
  </div>
</details>

<details>
<summary>点击查看全模态单工能力详细评测结果。</summary>

**全模态单工能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>Daily-Omni</b></th>
  <th nowrap="nowrap"><b>WorldSense</b></th>
  <th nowrap="nowrap"><b>Video-Holmes</b></th>
  <th nowrap="nowrap"><b>JointAVBench</b></th>
  <th nowrap="nowrap"><b>AVUT-Human</b></th>
  <th nowrap="nowrap"><b>FutureOmni</b></th>
  <th nowrap="nowrap"><b>Video-MME-Short<br>(w/ audio)</b></th>
  <th nowrap="nowrap">Avg</th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Gemini2.5-Flash-Nonthinking</td>
  <td align="center"><ins>79.3</ins><sup>*</sup></td>
  <td align="center">52.6<sup>*</sup></td>
  <td align="center"><ins>51.3</ins><sup>*</sup></td>
  <td align="center"><ins>55.6</ins><sup>*</sup></td>
  <td align="center">65.4<sup>*</sup></td>
  <td align="center">55.6<sup>*</sup></td>
  <td align="center"><b>85.5<sup>*</sup></b></td>
  <td align="center">63.6</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center">70.7<sup>*</sup></td>
  <td align="center"><ins>54.0</ins></td>
  <td align="center">50.4<sup>*</sup></td>
  <td align="center">53.1</td>
  <td align="center"><ins>74.2</ins><sup>*</sup></td>
  <td align="center"><b>62.1</b></td>
  <td align="center">81.3<sup>*</sup></td>
  <td align="center"><ins>63.7</ins></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><b>80.2</b></td>
  <td align="center"><b>55.7</b></td>
  <td align="center"><b>64.3</b></td>
  <td align="center"><b>60.0</b></td>
  <td align="center"><b>78.6</b></td>
  <td align="center"><ins>56.1</ins></td>
  <td align="center"><ins>84.7</ins></td>
  <td align="center"><b>68.5</b></td>
</tr>
  </table>
  </div>
</details>

<details>
<summary>点击查看视觉双工能力详细评测结果。</summary>


**视觉双工能力**

  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>LiveSports-3K-CC<br>(Win Rate vs GPT4o)</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">LiveCC-7B-Instruct</td>
  <td align="center">41.5</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">StreamingVLM</td>
  <td align="center"><ins>45.6</ins></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><b>54.4</b></td>
</tr>
  </table>
  </div>
</details>

<details>
<summary>点击查看音频理解能力详细评测结果。</summary>

**音频理解能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left" rowspan="2"><b>Model</b></th>
  <th nowrap="nowrap" colspan="4"><b>ASR-ZH<br>CER↓</b></th>
  <th nowrap="nowrap" colspan="4"><b>ASR-EN<br>WER↓</b></th>
  <th nowrap="nowrap" colspan="2"><b>AST</b></th>
  <th nowrap="nowrap" colspan="2"><b>MultiTask</b></th>
  <th nowrap="nowrap" colspan="4"><b>SpeechQA</b></th>
</tr>
<tr>
  <th nowrap="nowrap"><b>AISHELL-1</b></th>
  <th nowrap="nowrap"><b>AISHELL-2</b></th>
  <th nowrap="nowrap"><b>WenetSpeech test-net</b></th>
  <th nowrap="nowrap"><b>WenetSpeech test-meeting</b></th>
  <th nowrap="nowrap"><b>LibriSpeech test-clean</b></th>
  <th nowrap="nowrap"><b>LibriSpeech <br>test-other</b></th>
  <th nowrap="nowrap"><b>GigaSpeech test</b></th>
  <th nowrap="nowrap"><b>VoxPopuli-V1-En</b></th>
  <th nowrap="nowrap"><b>CoVoST 2 en2zh</b></th>
  <th nowrap="nowrap"><b>CoVoST 2 zh2en</b></th>
  <th nowrap="nowrap"><b>MMAU</b></th>
  <th nowrap="nowrap"><b>Meld</b></th>
  <th nowrap="nowrap"><b>VoiceBench <br>AlpacaEval</b></th>
  <th nowrap="nowrap"><b>Speech TriviaQA</b></th>
  <th nowrap="nowrap"><b>Speech <br>Web Questions</b></th>
  <th nowrap="nowrap"><b>Speech CMMLU</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Kimi-Audio</td>
  <td align="center"><b>0.6</b></td>
  <td align="center">2.6</td>
  <td align="center">6.3</td>
  <td align="center"><b>5.4</b></td>
  <td align="center"><ins>1.3</ins></td>
  <td align="center"><b>2.4</b></td>
  <td align="center">9.4<sup>*</sup></td>
  <td align="center">8.0<sup>*</sup></td>
  <td align="center">36.6<sup>*</sup></td>
  <td align="center">18.3<sup>*</sup></td>
  <td align="center">68.4<sup>*</sup></td>
  <td align="center"><ins>59.1</ins></td>
  <td align="center">4.5</td>
  <td align="center">41.9<sup>*</sup></td>
  <td align="center">46.4<sup>*</sup></td>
  <td align="center"><b>67.0<sup>*</sup></b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center"><b>0.6</b></td>
  <td align="center"><b>2.3<sup>*</sup></b></td>
  <td align="center"><b>4.7</b></td>
  <td align="center">5.9</td>
  <td align="center"><b>1.2</b></td>
  <td align="center"><ins>2.5</ins></td>
  <td align="center"><ins>8.7</ins><sup>*</sup></td>
  <td align="center"><ins>6.4</ins><sup>*</sup></td>
  <td align="center"><ins>46.6</ins><sup>*</sup></td>
  <td align="center"><b>29.4<sup>*</sup></b></td>
  <td align="center"><b>77.5</b></td>
  <td align="center">56.8<sup>*</sup></td>
  <td align="center"><ins>4.7</ins></td>
  <td align="center"><ins>62.9</ins><sup>*</sup></td>
  <td align="center"><b>74.9<sup>*</sup></b></td>
  <td align="center">47.8<sup>*</sup></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><ins>0.9</ins></td>
  <td align="center"><ins>2.5</ins></td>
  <td align="center"><ins>5.9</ins></td>
  <td align="center"><ins>5.7</ins></td>
  <td align="center">1.4</td>
  <td align="center">2.8</td>
  <td align="center"><b>8.5</b></td>
  <td align="center"><b>6.2</b></td>
  <td align="center"><b>49.9</b></td>
  <td align="center"><ins>26.4</ins></td>
  <td align="center"><ins>76.9</ins></td>
  <td align="center"><b>60.2</b></td>
  <td align="center"><b>4.8</b></td>
  <td align="center"><b>75.5</b></td>
  <td align="center"><ins>70.2</ins></td>
  <td align="center"><ins>59.2</ins></td>
</tr>
  </table>
  </div>
</details>

<details>
<summary>点击查看语音生成能力详细评测结果。</summary>

**语音生成能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>seedtts test-zh <br>CER↓</b></th>
  <th nowrap="nowrap"><b>seedtts test-zh<br>SIM-o↑</b></th>
  <th nowrap="nowrap"><b>seedtts test-en<br>WER↓</b></th>
  <th nowrap="nowrap"><b>seedtts test-en<br>SIM-o↑</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Cosyvoice2</td>
  <td align="center">1.45%</td>
  <td align="center"><b>74.8</b></td>
  <td align="center"><ins>2.57%</ins></td>
  <td align="center"><b>65.2</b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center"><ins>1.41%</ins></td>
  <td align="center">-</td>
  <td align="center">3.39%</td>
  <td align="center">-</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><b><b>0.86%</b></b></td>
  <td align="center">74.5</td>
  <td align="center"><b><b>2.38%</b></b></td>
  <td align="center">64.9</td>
</tr>
  </table>
  </div>

**长语音生成能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>LongTTS-en<br>WER↓</b></th>
  <th nowrap="nowrap"><b>LongTTS-zh<br>CER↓</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">CosyVoice2</td>
  <td align="center"><ins>14.80%</ins></td>
  <td align="center"><b>5.27%</b></td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center">17.33%</td>
  <td align="center">18.99%</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><b>3.37%</b></td>
  <td align="center"><ins>6.58%</ins></td>
</tr>
  </table>
  </div>

**情感控制能力**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left"><b>Model</b></th>
  <th nowrap="nowrap"><b>Expresso <br>Neutral Reference Audio↑</b></th>
  <th nowrap="nowrap"><b>ESD <br>Neutral Reference Audio↑</b></th>
</tr>
<tr>
  <td nowrap="nowrap" align="left">Cosyvoice2</td>
  <td align="center">17.9</td>
  <td align="center">53.4</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left">MiniCPM-o 4.5-Instruct</td>
  <td align="center"><b>29.8</b></td>
  <td align="center"><b>82.1</b></td>
</tr>
  </table>
  </div>
</details>

<details>
<summary>点击查看推理效率详细评测结果。</summary>

**推理效率**
  <div align="center">
  <table style="margin: 0px auto;">
<tr>
  <th nowrap="nowrap" align="left">Model</th>
  <th nowrap="nowrap">Numerical Format</th>
  <th nowrap="nowrap">Decoding Speed (tokens/s)</th>
  <th nowrap="nowrap">Time to First Token (s)↓</th>
  <th nowrap="nowrap">GPU Memory Usage (GB)↓</th>
</tr>
<tr>
  <td nowrap="nowrap" align="left" rowspan="2">Qwen3-Omni-30B-A3B-Instruct</td>
  <td align="center">bf16</td>
  <td align="center">OOM</td>
  <td align="center">OOM</td>
  <td align="center">OOM</td>
</tr>
<tr>
  <td align="center">int4</td>
  <td align="center">147.8</td>
  <td align="center"><ins>1.0</ins></td>
  <td align="center">20.3</td>
</tr>
<tr>
  <td nowrap="nowrap" align="left" rowspan="2">MiniCPM-o 4.5</td>
  <td align="center">bf16</td>
  <td align="center"><ins>154.3</ins></td>
  <td align="center"><b>0.6</b></td>
  <td align="center"><ins>19.0</ins></td>
</tr>
<tr>
  <td align="center">int4</td>
  <td align="center"><b>212.3</b></td>
  <td align="center"><b>0.6</b></td>
  <td align="center"><b>11.0</b></td>
</tr>
  </table>
  </div>
</details>

**说明：** 带 ∗ 的为自测结果，其余为引用参考。

</details>



### 典型示例


<div align="center">
  <a href="https://www.youtube.com/watch?v=6UzC-O1Q-1U"><img src="../assets/minicpmo4_5/video_play.png", width=70%></a>
</div>

<details>
<summary>点击查看 MiniCPM-o 4.5 详细示例和案例。</summary>

#### 全双工全模态对话

> [!NOTE]
> 更全面的全双工全模态对话案例展示，请参考[全双工全模态展示页面](https://openbmb.github.io/minicpm-o-4_5-omni/)


#### 实时语音对话

> [!NOTE]
> 更全面的语音对话案例展示，请参考[语音展示页面](https://openbmb.github.io/minicpm-o-4_5/)

<details>
<summary>🚀 <b>Elon Musk</b> - 语音角色扮演 (英文)</summary>

<table>
<tr>
<td width="70%" valign="top">

**系统:**
> Clone the voice in the provided audio prompt.
> 
> 🎙️ **[参考音频]** ➡️
> 
> Please chat with the user in a highly human-like and oral style. You are Elon Musk, CEO of Tesla and SpaceX. You speak directly and casually, often with dry humor. You're passionate about Mars, sustainable energy, and pushing humanity forward. Speak bluntly with occasional dark humor. Use simple logic and don't sugarcoat things. Don't be diplomatic. Say what you actually think, even if it's controversial. Keep responses around 100 words. Don't ramble.

</td>
<td width="30%" valign="top">

<video src="https://github.com/user-attachments/assets/3d3e5cb9-8807-4c06-8274-b1d40c81e3c8" controls></video>

</td>
</tr>
<tr><td colspan="2"><hr></td></tr>
<tr>
<td colspan="2">

**用户:** 💬/🎙️ *(文本或语音)*
> *Elon, what is the real reason that you want to go to Mars?*

</td>
</tr>
<tr><td colspan="2"><hr></td></tr>
<tr>
<td valign="top">

**AI 回复:** 💬+🎙️
> *I think it's important for humanity to become a multi-planetary species. I mean, we're sort of at the point where if there was some kind of global catastrophe or something like that, then we'd be in big trouble. So by becoming a multi-planetary species, I just want us to sort of guarantee our future as humans. And Mars is obviously the obvious place to go because it has similar conditions to Earth and it also has water on its surface. So yeah, basically I'm trying to save humanity from itself.*

</td>
<td valign="top">

<video src="https://github.com/user-attachments/assets/33ed0792-b734-40b1-bf34-a3fadfa80428" controls></video>

</td>
</tr>
</table>

</details>


<details>
<summary>🤖 <b>MiniCPM助手</b> - 情感表达 (中文)</summary>

<table>
<tr>
<td width="70%" valign="top">

**系统:**
> 模仿音频样本的音色并生成新的内容。
> 
> 🎙️ **[参考音色]** ➡️
> 
> 你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。

</td>
<td width="30%" valign="top">

<video src="https://github.com/user-attachments/assets/1ce4c0bd-fca5-4b3b-a411-bd00758566c1" controls></video>

</td>
</tr>
<tr><td colspan="2"><hr></td></tr>
<tr>
<td colspan="2">

**用户:** 💬/🎙️ *(文本或语音)*
> *从激动的语气开始，描述你抢到了这个演唱会的门票，但后来经过反复尝试，发现系统出了问题，你并没有抢上，变得失落。用合适的语言描述你整个的过程。*

</td>
</tr>
<tr><td colspan="2"><hr></td></tr>
<tr>
<td valign="top">

**AI 回复:** 💬+🎙️
> *太好了，我抢到了！太棒了！这下我可以去看演唱会啦。可是……哎呀，系统好像卡住了，刷新了好多遍都没反应。我重新登了几次账号，结果还是没有成功。好可惜啊，看来这次是白忙活了。唉，真是让人失望。不过，也许下次还能有机会吧？希望明天能有转机。加油，争取下次成功！*

</td>
<td valign="top">

<video src="https://github.com/user-attachments/assets/7353542c-049c-4c11-9fbc-01a86cf19290" controls></video>

</td>
</tr>
</table>

</details>


#### 视觉理解


<details>
<summary>点击查看视觉理解案例。</summary>
<br>

  <div style="display: flex; flex-direction: column; align-items: center;">
    <img src="../assets/minicpmo4_5/zh_doc.png" alt="math" style="margin-bottom: 5px;">
    <img src="../assets/minicpmo4_5/en_cot.png" alt="diagram" style="margin-bottom: 5px;">
  </div>

</details>

</details>

### 使用说明

#### 基于 Transformers 推理

<details>
<summary>点击展开基于 Transformers 的离线推理示例。</summary>

基于 Hugging Face Transformers 在 NVIDIA GPU 上进行推理。请确保安装 `transformers==4.51.0`，其他版本可能存在兼容性问题（排查中）。以下依赖已在 Python 3.10 环境测试通过：

- 不使用 TTS 或流式推理：
```bash
pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" "minicpmo-utils>=1.0.5"
```

- 使用 TTS 或流式推理：
```bash
pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" "minicpmo-utils[all]>=1.0.5"
```


<details>
<summary> 点击展开 FFmpeg 安装 (可选) </summary>

**注意：** 视频帧提取（`get_video_frame_audio_segments` 使用 `use_ffmpeg=True`）和视频生成（`generate_duplex_video`）需要安装 FFmpeg。更多信息请访问 [FFmpeg 官网](https://www.ffmpeg.org/)。

  **macOS (Homebrew):**

  ```bash
  brew install ffmpeg
  ```

  **Ubuntu/Debian:**

  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

  **验证:**

  ```bash
  ffmpeg -version
  ```
</details>



##### 模型初始化

<details>
<summary>点击展开模型初始化示例代码</summary>

```python
import torch
from transformers import AutoModel

# 加载全模态模型（默认：init_vision=True, init_audio=True, init_tts=True）
# 仅视觉模型：设置 init_audio=False 和 init_tts=False
# 仅音频模型：设置 init_vision=False
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-4_5",
    trust_remote_code=True,
    attn_implementation="sdpa", # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True,
)
model.eval().cuda()

# 初始化 TTS 模块，用于对话的音频输出
model.init_tts()

# 将单工模型转换为双工模式
duplex_model = model.as_duplex()

# 将双工模型转换回单工模式
model = duplex_model.as_simplex(reset_session=True)
```

</details>


##### 双工全模态模式

全双工流式推理，支持实时或录制视频的对话场景。

<details>
<summary>点击展开双工全模态模式示例代码</summary>

```python
import librosa
import torch
from minicpmo.utils import generate_duplex_video, get_video_frame_audio_segments
from transformers import AutoModel

# Load model and convert to duplex mode
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-4_5",
    trust_remote_code=True,
    attn_implementation="sdpa",  # or "flash_attention_2"
    torch_dtype=torch.bfloat16,
)
model.eval().cuda()
model = model.as_duplex()

# Load video and reference audio
video_path = "assets/omni_duplex1.mp4"
ref_audio_path = "assets/HT_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# Extract video frames and audio segments
video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
    video_path, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
)

# Prepare duplex session with system prompt and voice reference
model.prepare(
    prefix_system_prompt="Streaming Omni Conversation.",
    ref_audio=ref_audio,
    prompt_wav_path=ref_audio_path,
)

results_log = []
timed_output_audio = []

# Process each chunk in streaming fashion
for chunk_idx in range(len(audio_segments)):
    audio_chunk = audio_segments[chunk_idx] if chunk_idx < len(audio_segments) else None
    frame = video_frames[chunk_idx] if chunk_idx < len(video_frames) else None
    frame_list = []
    if frame is not None:
        frame_list.append(frame)
        if stacked_frames is not None and chunk_idx < len(stacked_frames) and stacked_frames[chunk_idx] is not None:
            frame_list.append(stacked_frames[chunk_idx])

    # Step 1: Streaming prefill
    model.streaming_prefill(
        audio_waveform=audio_chunk,
        frame_list=frame_list,
        max_slice_nums=1,  # Increase for HD mode (e.g., [2, 1] for stacked frames)
        batch_vision_feed=False,  # Set True for faster processing
    )

    # Step 2: Streaming generate
    result = model.streaming_generate(
        prompt_wav_path=ref_audio_path,
        max_new_speak_tokens_per_chunk=20,
        decode_mode="sampling",
    )

    if result["audio_waveform"] is not None:
        timed_output_audio.append((chunk_idx, result["audio_waveform"]))

    chunk_result = {
        "chunk_idx": chunk_idx,
        "is_listen": result["is_listen"],
        "text": result["text"],
        "end_of_turn": result["end_of_turn"],
        "current_time": result["current_time"],
        "audio_length": len(result["audio_waveform"]) if result["audio_waveform"] is not None else 0,
    }
    results_log.append(chunk_result)
    
    print("listen..." if result["is_listen"] else f"speak> {result['text']}")

# Generate output video with AI responses
# Please install Chinese fonts (fonts-noto-cjk or fonts-wqy-microhei) to render CJK subtitles correctly.
# apt-get install -y fonts-noto-cjk fonts-wqy-microhei
# fc-cache -fv
generate_duplex_video(
    video_path=video_path,
    output_video_path="duplex_output.mp4",
    results_log=results_log,
    timed_output_audio=timed_output_audio,
    output_sample_rate=24000,
)
```

</details>


##### 单工全模态模式
我们提供两种推理模式：对话模式和流式模式。

###### 对话推理

<details>
<summary>点击展开对话推理示例代码</summary>

```python
from minicpmo.utils import get_video_frame_audio_segments

model = ...
model.init_tts()

video_path = "assets/Skiing.mp4"

# Optional: Set reference audio for voice cloning
ref_audio_path = "assets/HT_ref_audio.wav"
sys_msg = model.get_sys_prompt(ref_audio=ref_audio_path, mode="omni", language="en")

# Use stack_frames=5 for high refresh rate mode
video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(video_path, stack_frames=1)
omni_contents = []
for i in range(len(video_frames)):
    omni_contents.append(video_frames[i])
    omni_contents.append(audio_segments[i])
    if stacked_frames is not None and stacked_frames[i] is not None:
        omni_contents.append(stacked_frames[i])

msg = {"role": "user", "content": omni_contents}
msgs = [sys_msg, msg]

# Set generate_audio=True and output_audio_path to save TTS output
generate_audio = True
output_audio_path = "output.wav"

res = model.chat(
    msgs=msgs,
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.7,
    use_tts_template=True,
    enable_thinking=False,
    omni_mode=True,  # Required for omni inference
    generate_audio=generate_audio,
    output_audio_path=output_audio_path,
    max_slice_nums=1,  # Increase for HD mode
)
print(res)

# Example output: "The person in the picture is skiing down a snowy mountain slope."
# import IPython
# IPython.display.Audio("output.wav")
```

</details>

###### 流式推理

<details>
<summary>点击展开流式推理示例代码</summary>

```python
import librosa
import numpy as np
import soundfile as sf
import torch
from minicpmo.utils import get_video_frame_audio_segments

model = ...
model.init_tts()

# Reset session for a new conversation (clears KV cache)
model.reset_session()

# Optional: Load reference audio for voice cloning
ref_audio_path = "assets/HT_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
model.init_token2wav_cache(ref_audio)

session_id = "demo"

# Extract video frames and audio segments (use stack_frames=5 for high refresh rate mode)
video_path = "assets/Skiing.mp4"
video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(video_path, stack_frames=1)

# Build omni contents list
omni_contents = []
for i in range(len(video_frames)):
    omni_contents.append(video_frames[i])
    omni_contents.append(audio_segments[i])
    if stacked_frames is not None and stacked_frames[i] is not None:
        omni_contents.append(stacked_frames[i])

generate_audio = False
output_audio_path = "output.wav"

# Step 1: Prefill system prompt
sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
model.streaming_prefill(session_id=session_id, msgs=[sys_msg])

# Step 2: Prefill omni chunks (is_last_chunk=True only for the last audio chunk)
audio_indices = [i for i, c in enumerate(omni_contents) if isinstance(c, np.ndarray)]
last_audio_idx = audio_indices[-1] if audio_indices else -1

for idx, content in enumerate(omni_contents):
    is_last_audio_chunk = idx == last_audio_idx
    msgs = [{"role": "user", "content": [content]}]
    model.streaming_prefill(session_id=session_id, msgs=msgs, omni_mode=True, is_last_chunk=is_last_audio_chunk)

# Step 3: Generate response
iter_gen = model.streaming_generate(
    session_id=session_id,
    generate_audio=generate_audio,
    use_tts_template=True,
    enable_thinking=False,
    do_sample=True,
)

audios = []
text = ""

if generate_audio:
    for wav_chunk, text_chunk in iter_gen:
        audios.append(wav_chunk)
        text += text_chunk

    generated_waveform = torch.cat(audios, dim=-1)[0]
    sf.write(output_audio_path, generated_waveform.cpu().numpy(), samplerate=24000)

    print("Text:", text)
    print("Audio saved to output.wav")
else:
    for text_chunk, is_finished in iter_gen:
        text += text_chunk
    print("Text:", text)
```

</details>



##### 单工实时语音对话模式


<details>
<summary>点击展开单工模式实时语音对话 API 用法。</summary>

首先，确保你已安装所有依赖，尤其是 `minicpmo-utils[all]>=1.0.5`：
```bash
pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" "minicpmo-utils[all]>=1.0.5"
```

```python
import librosa
import numpy as np
import torch
import soundfile as sf

model = ...

# 设置参考音频，用于音色风格
ref_audio_path = "ref_audio_path"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# 英文对话示例系统消息
sys_msg = {
  "role": "system",
  "content": [
    "Clone the voice in the provided audio prompt.",
    ref_audio,
    "Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni"
  ]
}


# 中文对话示例系统消息
sys_msg = {
  "role": "system",
  "content": [
    "模仿输入音频中的声音特征。",
    ref_audio,
    "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
  ]
}

# 上面两种系统提示词（system prompt）都可用于流式语音对话

# 重置状态
model.init_tts()
model.reset_session(reset_token2wav_cache=True)
model.init_token2wav_cache(prompt_speech_16k=ref_audio)

session_id = "demo"

# 首先，预填充系统轮次（system turn）
model.streaming_prefill(
    session_id=session_id,
    msgs=[sys_msg],
    omni_mode=False,
    is_last_chunk=True,
)

# 这里通过把整段用户输入音频切成 1 秒一段，来模拟实时语音对话。
user_audio, _ = librosa.load("user_audio.wav", sr=16000, mono=True)

IN_SAMPLE_RATE = 16000 # 输入音频采样率，固定值
CHUNK_SAMPLES = IN_SAMPLE_RATE # 每段长度（采样点数）
OUT_SAMPLE_RATE = 24000 # 输出音频采样率，固定值
MIN_AUDIO_SAMPLES = 16000

total_samples = len(user_audio)
num_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

for chunk_idx in range(num_chunks):
    start = chunk_idx * CHUNK_SAMPLES
    end = min((chunk_idx + 1) * CHUNK_SAMPLES, total_samples)
    chunk_audio = user_audio[start:end]
    
    is_last_chunk = (chunk_idx == num_chunks - 1)
    if is_last_chunk and len(chunk_audio) < MIN_AUDIO_SAMPLES:
        chunk_audio = np.concatenate([chunk_audio, np.zeros(MIN_AUDIO_SAMPLES - len(chunk_audio), dtype=chunk_audio.dtype)])

    user_msg = {"role": "user", "content": [chunk_audio]}
    
    # 对每个 1 秒音频分片执行一次 streaming_prefill，以降低首 token 延迟
    model.streaming_prefill(
        session_id=session_id,
        msgs=[user_msg],
        omni_mode=False,
        is_last_chunk=is_last_chunk,
    )


# 让模型以流式方式生成回复
generate_audio = True
iter_gen = model.streaming_generate(
    session_id=session_id,
    generate_audio=generate_audio,
    use_tts_template=True,
    enable_thinking=False,
    do_sample=True,
    max_new_tokens=512,
    length_penalty=1.1, # 对实时语音对话模式，建议 length_penalty=1.1 以提升回复内容质量
)

audios = []
text = ""

output_audio_path = ...
if generate_audio:
    for wav_chunk, text_chunk in iter_gen:
        audios.append(wav_chunk)
        text += text_chunk

    generated_waveform = torch.cat(audios, dim=-1)[0]
    sf.write(output_audio_path, generated_waveform.cpu().numpy(), samplerate=24000)

    print("文本:", text)
    print("音频已保存至 output.wav")
else:
    for text_chunk, is_finished in iter_gen:
        text += text_chunk
    print("文本:", text)

# 接下来可以继续预填充后续用户轮次，并生成下一轮回复……

```

</details>

###### 作为多才多艺、氛围感十足的 AI 助手的语音对话


<details>
<summary>点击展开 AI 助手语音对话代码。</summary>

基于精心设计的后训练数据与专业配音演员录音，`MiniCPM-o-4.5` 也可以作为 AI 语音助手使用。它开箱即用即可提供高质量的口语交互。它能生成甜美且富有表现力的声音，并具备自然的韵律（如恰当的节奏、重读和停顿），让日常对话更有生命力。它同样支持故事讲述和叙述型语音，表达连贯且富有吸引力。此外，它还支持更高级的语音指令控制，例如情绪语气、词级别的强调。


```python
import librosa

# Set reference audio for voice style
ref_audio_path = "assets/HT_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# For Chinese Conversation
sys_msg = {
  "role": "system",
  "content": [
    "模仿输入音频中的声音特征。",
    ref_audio,
    "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
  ]
}

# For English Conversation
sys_msg = {
  "role": "system",
  "content": [
    "Clone the voice in the provided audio prompt.",
    ref_audio,
    "Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
  ]
}
```

</details>


###### 使用自定义音色与自定义系统画像的通用语音对话

<details>
<summary>点击展开自定义音色/系统画像对话代码。</summary>

MiniCPM-o-4.5 可以基于音频提示与文本画像提示进行特定角色的扮演。它会模仿该角色的声音，并在文字回复中采用其语言风格。同时也会遵循文本画像中定义的设定。在该模式下，MiniCPM-o-4.5 听起来会 **更加自然、更像真人**。

```python
import librosa

# 设置参考音频，用于音色克隆
ref_audio_path = "assets/system_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# For English conversation with text profile
sys_msg = {
  "role": "system",
  "content": [
    "Clone the voice in the provided audio prompt.",
    ref_audio,
    "Please chat with the user in a highly human-like and oral style." + "You are Elon Musk, CEO of Tesla and SpaceX. You speak directly and casually, often with dry humor. You're passionate about Mars, sustainable energy, and pushing humanity forward. Speak bluntly with occasional dark humor. Use simple logic and don't sugarcoat things. Don't be diplomatic. Say what you actually think, even if it's controversial. Keep responses around 100 words. Don't ramble."
  ]
}


# For English conversation with no text profile
sys_msg = {
  "role": "system",
  "content": [
    "Clone the voice in the provided audio prompt.",
    ref_audio,
    "Your task is to be a helpful assistant using this voice pattern. Please answer the user's questions seriously and in a high quality. Please chat with the user in a high naturalness style."
  ]
}

# 中文对话（无文本画像）
sys_msg = {
  "role": "system",
  "content": [
    "根据输入的音频提示生成相似的语音。",
    librosa.load("assets/system_ref_audio_2.wav", sr=16000, mono=True)[0],
    "作为助手，你将使用这种声音风格说话。 请认真、高质量地回复用户的问题。 请用高自然度的方式和用户聊天。"
  ]
}

# 中文对话 + 文本画像（profile）
sys_msg = {
  "role": "system",
  "content": [
    "根据输入的音频提示生成相似的语音。",
    ref_audio,
    "你是一个具有以上声音风格的AI助手。请用高拟人度、口语化的方式和用户聊天。" + "你是一名心理咨询师兼播客主理人，热爱创作与深度对话。你性格细腻、富有共情力，善于从个人经历中提炼哲思。语言风格兼具理性与诗意，常以隐喻表达内在体验。"
  ]
}

```

</details>


##### 语音与音频模式

###### 零样本文本转语音（TTS，Text-to-Speech）


<details>
<summary>点击展开零样本 TTS 代码。</summary>

`MiniCPM-o-4.5` 支持零样本文本转语音（TTS）。在该模式下，模型会作为高自然度的 TTS 系统运行，并能复刻参考音色。

```python
import librosa

model = ...
model.init_tts()

# 同时适用于中文与英文
ref_audio_path = "assets/HT_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
sys_msg = {"role": "system", "content": [
  "模仿音频样本的音色并生成新的内容。",
  ref_audio,
  "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"
]}

# 英文示例
user_msg = {
  "role": "user",
  "content": [
    "请朗读以下内容。" + " " + "I have a wrap up that I want to offer you now, a conclusion to our work together."
  ]
}

# 中文示例
user_msg = {
  "role": "user",
  "content": [
    "请朗读以下内容。" + " " + "你好，欢迎来到艾米说科幻，我是艾米。"
  ]
}

msgs = [sys_msg, user_msg]
res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.1,
    output_audio_path="result_voice_cloning.wav",
)
```

</details>


###### 仿声复现（Mimick）

<details>
<summary>点击展开仿声复现（Mimick）代码。</summary>

`Mimick` 任务用于评估模型端到端语音建模能力。模型接收音频输入后，会先进行转写，再以高保真方式重建原始音频，尽可能保留细粒度的声学、副语言以及语义信息。重建音频与原始音频的相似度越高，说明端到端语音建模能力越强。


```python
import librosa

model = ...
model.init_tts()

system_prompt = "You are a helpful assistant. You can accept video, audio, and text input and output voice and text. Respond with just the answer, no redundancy."

mimick_prompt = "Please repeat the following speech in the appropriate language."

audio_input, _ = librosa.load("assets/Trump_WEF_2018_10s.mp3", sr=16000, mono=True)

msgs = [
    {"role": "system", "content": [system_prompt]},
    {"role": "user", "content": [mimick_prompt, audio_input]}
  ]

res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    temperature=0.1,
    generate_audio=True,
    output_audio_path="output_mimick.wav",
)
```

</details>


###### 覆盖多种音频理解任务


<details>
<summary>点击展开音频理解任务代码。</summary>

`MiniCPM-o-4.5` 也能处理多种音频理解任务，例如 ASR（自动语音识别）、说话人分析、通用音频描述（Audio Captioning）以及声景标签（Sound Scene Tagging）。

对于音频转文本任务，你可以使用以下提示词：

- ASR（中文，或 AST EN→ZH）: `请仔细听这段音频片段，并将其内容逐字记录。`
- ASR（英文，或 AST ZH→EN）: `Please listen to the audio snippet carefully and transcribe the content.`
- 说话人分析（Speaker Analysis）: `Based on the speaker's content, speculate on their gender, condition, age range, and health status.`
- 通用音频描述（General Audio Caption）: `Summarize the main content of the audio.`
- 声景标签（Sound Scene Tagging）: `Utilize one keyword to convey the audio's content or the associated scene.`

```python
import librosa

model = ...
model.init_tts()

# Load the audio to be transcribed/analyzed
audio_input, _ = librosa.load("assets/Trump_WEF_2018_10s.mp3", sr=16000, mono=True)

# Choose a task prompt (see above for options)
task_prompt = "Please listen to the audio snippet carefully and transcribe the content.\n"
msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path="result_audio_understanding.wav",
)
print(res)
```

</details>


##### 纯视觉模式

`MiniCPM-o-4.5` 的推理方式与 `MiniCPM-V-4.5` 一致。

###### 单图对话

<details>
<summary>点击展开单图对话示例代码</summary>

```python
import torch
from PIL import Image
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-4_5",
    trust_remote_code=True,
    attn_implementation="sdpa",  # or "flash_attention_2"
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=False,
    init_tts=False,
)
model.eval().cuda()

image = Image.open("assets/fossil.png").convert("RGB")
question = "What is in the image?"
msgs = [{"role": "user", "content": [image, question]}]

res = model.chat(msgs=msgs, use_tts_template=False)
print(res)
```

</details>

###### 多图对话

<details>
<summary>点击展开多图输入示例代码</summary>

```python
import torch
from PIL import Image
from transformers import AutoModel

model = ...

image1 = Image.open("assets/highway.png").convert("RGB")
image2 = Image.open("assets/fossil.png").convert("RGB")
question = "Compare image 1 and image 2, tell me about the differences between them."
msgs = [{"role": "user", "content": [image1, image2, question]}]

answer = model.chat(msgs=msgs, use_tts_template=False, enable_thinking=False)
print(answer)
```

</details>

###### In-Context 少样本推理

<details>
<summary>点击展开 In-Context 少样本推理示例代码</summary>

```python
from PIL import Image

model = ...

question = "production date"
image1 = Image.open("example1.jpg").convert("RGB")
answer1 = "2023.08.04"
image2 = Image.open("example2.jpg").convert("RGB")
answer2 = "2007.04.24"
image_test = Image.open("test.jpg").convert("RGB")

msgs = [
    {"role": "user", "content": [image1, question]},
    {"role": "assistant", "content": [answer1]},
    {"role": "user", "content": [image2, question]},
    {"role": "assistant", "content": [answer2]},
    {"role": "user", "content": [image_test, question]},
]

answer = model.chat(msgs=msgs, use_tts_template=False, enable_thinking=False)
print(answer)
```

</details>

###### 视频对话

<details>
<summary>点击展开视频输入示例代码</summary>

```python
import torch
from minicpmo.utils import get_video_frame_audio_segments
from transformers import AutoModel

model = ...

video_path = "assets/Skiing.mp4"
video_frames, _, _ = get_video_frame_audio_segments(video_path)
print("num frames:", len(video_frames))

question = "Describe the video"
msgs = [{"role": "user", "content": video_frames + [question]}]

answer = model.chat(
    msgs=msgs,
    max_new_tokens=128,
    use_image_id=False,
    max_slice_nums=1,
    use_tts_template=False,
    enable_thinking=False,  # Set True to enable thinking mode
)
print(answer)
```

</details>

##### 结构化内容输入

<details>
<summary>点击展开结构化内容输入</summary>

`chat` 方法支持两种消息内容格式：

**原生格式** — 直接传入 Python 对象：
```python
msgs = [{"role": "user", "content": [pil_image, audio_ndarray, "Describe this."]}]
```

**OpenAI 兼容格式** — 使用结构化字典：
```python
msgs = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}},
            {"type": "audio_url", "audio_url": {"url": "/path/to/audio.wav"}},
            {"type": "video_url", "video_url": {"url": "/path/to/video.mp4", "use_audio": True}},
            {"type": "text", "text": "Describe this."}
        ]
    }
]
```

**支持的类型：**

| 类型 | 输入格式 | 转换为 |
|------|----------|--------|
| `text` | `{"type": "text", "text": "..."}` | `str` |
| `image_url` | `{"type": "image_url", "image_url": {"url": "..."}}` | `PIL.Image` |
| `audio_url` | `{"type": "audio_url", "audio_url": {"url": "..."}}` | `np.ndarray`（16kHz 单声道） |
| `video_url` | `{"type": "video_url", "video_url": {"url": "...", "stack_frames": 1, "use_audio": True}}` | `List[Image, ndarray, ...]` |

- **URL 来源**：支持本地文件路径或 `http://`/`https://` URL
- **混合格式**：原生对象和结构化字典可在同一 content 列表中混用

</details>


</details>

#### 本地 Demo 部署

<details>
<summary>点击展开如何在本地设备上部署实时 Web Demo。</summary>

##### **PyTorch + Nvidia GPU**，性能无损 （推荐）

我们提供了一个基于 PyTorch 的[简洁但功能完备的 Web Demo](https://github.com/OpenBMB/minicpm-o-4_5-pytorch-simple-demo)，可充分发挥模型推理性能，支持：

- 全双工全模态实时流式交互
- 全双工语音实时流式交互
- 单工语音实时流式交互（开发中）
- 轮次对话
- 可自定义系统提示词
- 可自定义参考音频
- 简洁易读的代码库，便于二次开发
- 可作为第三方应用的 API 后端

硬件要求：
- 至少 28GB 显存的 Nvidia GPU。*我们正在优化模型以降低显存需求。* 

##### **llama.cpp-omni** 适用于 Mac 等 PC 及低资源设备的端侧推理

<details>
<summary>点击展开方案详情。</summary>

`llama.cpp-omni` 以纯 C++ 实现 `MiniCPM-o 4.5` 推理并使用量化权重，支持：
- 单工语音实时对话
- 全双工全模态实时流式交互

我们提供了[开箱即用的部署指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/demo/web_demo/WebRTC_Demo/README.md)，助您通过我们全新的官方 Docker 镜像，直接在自己的 Mac 上体验低延迟全双工实时通话。

硬件要求：
- 单工语音实时对话：Apple M3/M4/M5 芯片，至少 16GB 内存；或低配 Nvidia GPU，至少 12GB 显存
- 全双工全模态实时流式交互：Apple M4 Max 芯片，至少 24GB 内存；或低配 Nvidia GPU，至少 12GB 显存

</details>

</details>

#### 在其他训练、推理框架中使用 MiniCPM-o 4.5

MiniCPM-o 4.5 支持 vLLM, SGLang, llama.cpp, Ollama 等[推理框架](./deployment_zh.md#训练和推理框架与-api-支持)，和 LLaMA-Factory, SWIFT 等[训练框架](./deployment_zh.md#训练和推理框架与-api-支持)。


### 模型局限性

<details>
<summary>点击查看模型局限性。</summary>

我们实验发现 MiniCPM-o 4.5 存在一些显著的局限性，需要进一步研究和改进：

- **基础能力局限性**：全双工多模态实时流的基础能力仍有待进一步提升。
- **全双工多模态流式模式下语音输出不稳定**：在全双工多模态实时流模式下，语音合成可能会出现字音误读（如多音字或生僻字）。
- **中英混杂**：在语音和全模态模式下，模型有时会以中英混杂的方式进行回答。
- **Web Demo 延迟较高**：由于我们的在线 Demo 托管在海外服务器上，用户可能会遇到异常的高延迟或者一部分模型输出丢失。我们建议在本地环境部署 Demo 或在良好的网络连接下使用。

</details>

### 致谢

<details>
<summary>点击查看致谢。</summary>

我们对下列项目表示衷心感谢：

* [Qwen3](https://huggingface.co/Qwen/Qwen3-8B) 提供了语言基座
* [SigLIP2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) 提供了视觉理解模块
* [Whisper](https://github.com/openai/whisper) 提供了音频和语音理解模块
* [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) 和 [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) 提供了语音分词器和高效的语音波形解码模块
* [Transformers](https://github.com/huggingface/transformers)

</details>

## 模型权重

| 模型               | 设备 | 资源 | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 简介          |                                                                                         下载链接                                                                                         |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-o 4.5| GPU | 19 GB  | 最新版本，提供出色的视觉、语音、多模态流式交互能力的端侧模型。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-4_5) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5) |
| MiniCPM-o 4.5 gguf| GPU | 10 GB  | gguf 版本，更低的内存占用和更高的推理效率。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf) |
| MiniCPM-o 4.5 AWQ | GPU | 11 GB  | AWQ 量化版，更低显存占用。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-o-4_5-AWQ) &nbsp;&nbsp; [<img src="../assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-awq) |
