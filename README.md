# MiniCPM-V

<div align="center">
<img src="./assets/minicpm_v_and_minicpm_o_title.png" width="500em" ></img>
</div>

**A GPT-4o Level MLLM for Single Image, Multi Image and Video Understanding on Your Phone**

<strong>
<a href="./README_zh.md">中文</a> | English
</strong>

<div style="display: inline-flex; align-items: center; margin-right: 10px;">
<img src="./assets/wechat.png" alt="WeChat" style="margin-right: 4px;">
<a href="docs/wechat.md" target="_blank">WeChat</a>
</div>
|
<div style="display: inline-flex; align-items: center; margin-left: 10px;">
<img src="./assets/discord.png" alt="Discord" style="margin-right: 4px;">
<a href="https://discord.gg/rftuRMbqzf" target="_blank">Discord</a>
</div>

<p align="center">
   MiniCPM-V 4.5 <a href="https://huggingface.co/openbmb/MiniCPM-V-4_5">🤗</a> <a href="http://101.126.42.235:30910/">🤖</a> | MiniCPM-o 2.6 <a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">🤗</a> <a href="https://minicpm-omni-webdemo-us.modelbest.cn/"> 🤖</a> | <a href="https://github.com/OpenSQZ/MiniCPM-V-Cookbook">🍳 Cookbook</a> | 
  📄 Technical Report (Coming Soon)
</p>

</div>

**MiniCPM-V** is a series of efficient end-side multimodal LLMs (MLLMs), which accept images, videos and text as inputs and deliver high-quality text outputs. **MiniCPM-o** additionally takes audio as inputs and provide high-quality speech outputs in an end-to-end fashion. Since February 2024, we have released 7 versions of the model, aiming to achieve **strong performance and efficient deployment**. The most notable models in the series currently include:

- **MiniCPM-V 4.5**: 🔥🔥🔥 The latest and most capable model in the MiniCPM-V series. With a total of 8B parameters, this model **outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B** in vision-language capabilities, making it the most performant on-device multimodal model in the open-source community. This version brings **new features including efficient high refresh rate and long video understanding (up to 96x compression rate for video tokens), controllable hybrid fast/deep thinking, strong handwritten OCR and complex table/document parsing**. It also advances MiniCPM-V's popular features such as trustworthy behavior, multilingual support and end-side deployability.

- **MiniCPM-o 2.6**: ⭐️⭐️⭐️ The most capable model in the MiniCPM-o series. With a total of 8B parameters, this end-to-end model **achieves comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming**, making it one of the most versatile and performant models in the open-source community. For the new voice mode, MiniCPM-o 2.6 **supports bilingual real-time speech conversation with configurable voices**, and also allows for fun capabilities such as emotion/speed/style control, end-to-end voice cloning, role play, etc. Due to its superior token density, MiniCPM-o 2.6 can for the first time **support multimodal live streaming on end-side devices** such as iPad.

- **MiniCPM-V 4.0**: 🚀🚀🚀 The latest efficient model in the MiniCPM-V series. With a total of 4B parameters, the model **surpasses GPT-4.1-mini-20250414, Qwen2.5-VL-3B-Instruct, and InternVL2.5-8B** in image understanding on the OpenCompass evaluation. With its small parameter-size and efficient architecture, MiniCPM-V 4.0 is an ideal choice for on-device deployment on the phone (e.g., **less than 2s first token delay and more than 17 token/s decoding** on iPhone 16 Pro Max using the open-sourced iOS App).

- **MiniCPM-V 2.6**: The most capable model in the MiniCPM-V series. With a total of 8B parameters, the model surpasses GPT-4V in single-image, multi-image and video understanding. It outperforms GPT-4o mini, Gemini 1.5 Pro and Claude 3.5 Sonnet in single image understanding, and can for the first time support real-time video understanding on iPad.

## Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)

## 🌟 Star History <!-- omit in toc -->

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=OpenBMB/MiniCPM-V&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=OpenBMB/MiniCPM-V&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=OpenBMB/MiniCPM-V&type=Date"
  />
</picture>

## Key Techniques and Other Multimodal Projects <!-- omit in toc -->

👏 Welcome to explore key techniques of MiniCPM-o/V and other multimodal projects of our team:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLPR](https://github.com/OpenBMB/RLPR) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Citation <!-- omit in toc -->

If you find our model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```
