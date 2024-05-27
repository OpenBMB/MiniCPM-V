## OmniLMM-12B

> OmniLMM-12B is released at early time of this project. We recommond you to use our [recently released models](./README_en.md), for better performance and efficiency.

> Archieve at: 2024-05-19


**OmniLMM-12B** is the most capable version. The model is built based on EVA02-5B and Zephyr-7B-β, connected with a perceiver resampler layer, and trained on multimodal data in a curriculum fashion. The model has three notable features:

- 🔥 **Strong Performance.** 

  OmniLMM-12B achieves **leading performance** among models with comparable sizes, surpassing established LMMs on multiple benchmarks (including MME, MMBench, SEED-Bench, etc). The model also endows rich multi-modal world knowledge.

- 🏆 **Trustworthy Behavior.** 

  LMMs are known for suffering from hallucination, often generating text that is not factually grounded in images (e.g., faithfully describing non-existing objects in images). OmniLMM-12B is **the first state-of-the-art open-source LMM aligned via multimodal RLHF for trustworthy behavior** (using the recent [RLHF-V](https://rlhf-v.github.io/) technique). It **ranks #1** among open-source models on [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench), and **outperforms GPT-4V** on [Object HalBench](https://arxiv.org/abs/2312.00849).

- 🕹 **Real-time Multimodal Interaction.** 

  We combine the OmniLMM-12B and GPT-3.5 (text-only) into a **real-time multimodal interactive assistant**. The assistant accepts video streams from the camera and speech streams from the microphone and emits speech output. While still primary, we find the model can **replicate some of the fun cases shown in the Gemini Demo video, without any video edition**.


### Evaluation <!-- omit in toc -->
<div align="center">
    <img src=assets/radar_omnilmm12b.png width=66% />
</div>
<details>
<summary>Click to view results on MME, MMBench, MMMU, MMBench, MMHal-Bench, Object HalBench, SeedBench, LLaVA Bench, MathVista. </summary>

<table>
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th>MME</th>
    <th nowrap="nowrap">MMB dev (en)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >MMHal-Bench</th>
    <th nowrap="nowrap" >Object HalBench</th>
    <th nowrap="nowrap" >SeedBench-I</th>
    <th>MathVista</th>
    <th nowrap="nowrap" >LLaVA Bench</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">GPT-4V†</td>
    <td>-</td>
    <td>1771.5</td>
    <td>75.1 </td>
    <td>56.8</td>
    <td>3.53 / 70.8</td>
    <td>86.4 / 92.7</td>
    <td>71.6 </td>
    <td>47.8 </td>
    <td>93.1 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">Qwen-VL-Plus†</td>
    <td>-</td>
    <td>2183.4</td>
    <td>66.2 </td>
    <td>45.2</td>
    <td>- </td>
    <td>- </td>
    <td>65.7 </td>
    <td>36.0 </td>
    <td>73.7 </td>
  </tr>
  <tr>
    <td align="left">Yi-VL 6B</td>
    <td align="right">6.7B </td>
    <td>1915.1 </td>
    <td>68.6 </td>
    <td>40.3 </td>
    <td>- </td>
    <td>- </td>
    <td>67.5 </td>
    <td>28.8 </td>
    <td>51.9 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right">9.6B</td>
    <td>1860.0</td>
    <td>60.6 </td>
    <td>35.9</td>
    <td>2.93 / 59.4</td>
    <td>56.2 / 80.0</td>
    <td>64.8 </td>
    <td>33.8 </td>
    <td>67.7 </td>
  </tr>
  <tr>
    <td align="left" >CogVLM-Chat</td>
    <td align="right">17.4B</td>
    <td>1736.6</td>
    <td>63.7 </td>
    <td>32.1 </td>
    <td>2.68 / 52.1 </td>
    <td>73.6 / 87.4 </td>
    <td>68.8 </td>
    <td>34.7 </td>
    <td>73.9 </td>
  </tr>
  <tr>
    <td align="left" >LLaVA 1.5</td>
    <td align="right">13.6B </td>
    <td>1808.4 </td>
    <td>68.2 </td>
    <td>36.4 </td>
    <td>2.71 / 51.0 </td>
    <td>53.7 / 77.4 </td>
    <td>68.1 </td>
    <td>26.4 </td>
    <td>64.6 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>OmniLMM-12B</b></td>
    <td align="right">11.6B </td>
    <td>1935.8 </td>
    <td>71.6 </td>
    <td>40.7 </td>
    <td>3.45 / 68.8 </td>
    <td>90.3 / 95.5 </td>
    <td>71.1 </td>
    <td>34.9 </td>
    <td>72.0 </td>
  </tr>
</tbody>
</table>
<small>†: Proprietary models</small>
<br>
</details>

### Examples <!-- omit in toc -->

<table align="center" >
  <p align="center" > 
    <img src="assets/omnilmm-12b-examples_2.png" />
  </p>
</table>


We combine the OmniLMM-12B and GPT-3.5 (text-only) into a **real-time multimodal interactive assistant**. Video frames are described in text using OmniLMM-12B, and ChatGPT 3.5 (text-only) is employed to generate response according to the descriptions and user prompts. The demo video is a raw recording without edition. 

<div align="center" >
  <video controls src="https://github.com/OpenBMB/OmniLMM/assets/157115220/485a8f52-fb4d-4eca-8fee-506347efcfc6" type="video/mp4" width=80%/>
</div>

### Model Zoo

| Model                | Description       | Download Link |
|:----------------------|:-------------------|:---------------:|
| OmniLMM-12B | The most capable version with leading performance.   |  [🤗](https://huggingface.co/openbmb/OmniLMM-12B) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/OmniLMM-12B/files) |
