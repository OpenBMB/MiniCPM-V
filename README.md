# MiniCPM-o

<div align="center">
<img src="./assets/MiniCPM-o.png" width="300em" >
</div>

**A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone**

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
MiniCPM-V 4.0 <a href="https://huggingface.co/openbmb/MiniCPM-V-4">🤗</a> <a href="https://minicpm-v.openbmb.cn/">🤖</a> | MiniCPM-o 2.6 <a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">🤗</a> <a href="https://minicpm-omni-webdemo-us.modelbest.cn/">🤖</a> | MiniCPM-V 2.6 <a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">🤗</a> <a href="http://120.92.209.146:8887/">🤖</a> | <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook">🍳 Cookbook</a> | 📄 Technical Blog [<a href="https://openbmb.notion.site/01c4f84f">English</a>/<a href="https://openbmb.notion.site/d37c418c">中文</a>]
</p>

**MiniCPM-o** is the latest series of end-side multimodal LLMs (MLLMs) ungraded from MiniCPM-V. The models can now take images, video, text, and audio as inputs and provide high-quality text and speech outputs in an end-to-end fashion. Since February 2024, we have released 6 versions of the model, aiming to achieve **strong performance and efficient deployment**. The most notable models in the series currently include:

- **MiniCPM-V 4.0**: 🚀🚀🚀 The latest efficient model in the MiniCPM-V series. With a total of 4B parameters, the model **surpasses GPT-4.1-mini-20250414, Qwen2.5-VL-3B-Instruct, and InternVL2.5-8B** in image understanding on the OpenCompass evaluation. With its small parameter-size and efficient architecture, MiniCPM-V 4.0 is an ideal choice for on-device deployment on the phone (e.g., **less than 2s first token delay and more than 17 token/s decoding** on iPhone 16 Pro Max using the open-sourced iOS App).

- **MiniCPM-o 2.6**: 🔥🔥🔥 The most capable model in the MiniCPM-o series. With a total of 8B parameters, this end-to-end model **achieves comparable performance to GPT-4o-202405 in vision, speech, and multimodal live streaming**, making it one of the most versatile and performant models in the open-source community. For the new voice mode, MiniCPM-o 2.6 **supports bilingual real-time speech conversation with configurable voices**, and also allows for fun capabilities such as emotion/speed/style control, end-to-end voice cloning, role play, etc. It also advances MiniCPM-V 2.6's visual capabilities such **strong OCR capability, trustworthy behavior, multilingual support, and video understanding**. Due to its superior token density, MiniCPM-o 2.6 can for the first time **support multimodal live streaming on end-side devices** such as iPad.

- **MiniCPM-V 2.6**: The most capable model in the MiniCPM-V series. With a total of 8B parameters, the model surpasses GPT-4V in single-image, multi-image and video understanding. It outperforms GPT-4o mini, Gemini 1.5 Pro and Claude 3.5 Sonnet in single image understanding, and can for the first time support real-time video understanding on iPad.
