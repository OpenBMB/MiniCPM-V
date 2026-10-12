[← 首页](../README_zh.md) · [English](./changelog_en.md)

# 更新日志

按日期倒序保留模型发布与项目更新。历史消息中的框架支持状态对应消息发布时点；当前使用方式请以模型与部署文档为准。

## 2026

* [2026.10.12] 🔥🔥🔥 我们开源了 MiniCPM-V 4.7，首个端侧无损流畅运行的开源 MoE 模型，刷新多模态性能—效率帕累托前沿。欢迎试用！

* [2026.06.25] 🔥🔥🔥 MiniCPM-V 4.6 已正式合并至 Ollama 官方模型库，为 Ollama 用户带来高效的图像与视频理解能力。欢迎[体验](https://ollama.com/library/minicpm-v4.6)！

* [2026.05.17] ⭐️⭐️⭐️ 我们发布了 MiniCPM-V 4.6 和 MiniCPM-o 4.5 的 [API 接口](./api.md)。

* [2026.05.11] 🔥🔥🔥 我们开源了 MiniCPM-V 4.6，支持 4倍/16倍 混合视觉 token 压缩率，凭借出色的编码效率和 1.3B 的轻量规模，它是我们端侧部署最友好的一代模型，高并发场景 token 吞吐达到 Qwen3.5 0.8B 的 ~1.5 倍。欢迎试用！

* [2026.05.07] 📢📢📢 我们发布了 MiniCPM-o 4.5 技术报告，介绍了其实现实时全双工全模态交互的关键技术。欢迎点击[这里](https://huggingface.co/papers/2604.27393)查看。

* [2026.02.06] 🥳 🥳 🥳 我们开源了可在 Mac 或 GPU 等本地设备上部署的实时 Web Demo。[立即体验](./minicpm_o4dot5_zh.md#本地-demo-部署)！

* [2026.02.05] 📢📢📢 我们注意到，由于网络状况原因，网页版演示可能会出现显著的延迟问题。我们正在积极工作，将尽快提供实时交互演示版的Docker镜像供本地部署，敬请持续关注！

* [2026.02.03] 🔥🔥🔥 我们开源了 MiniCPM-o 4.5，该模型视觉和语音能力达到了 Gemini 2.5 Flash 水平，同时支持全双工多模态流式交互。欢迎试用！

## 2025

* [2025.09.18] 📢📢📢 MiniCPM-V 4.5 技术报告已发布! 欢迎点击[这里](https://arxiv.org/abs/2509.18154)查看.

* [2025.09.01] ⭐️⭐️⭐️ MiniCPM-V 4.5 已被 [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15575)、[vLLM](https://github.com/vllm-project/vllm/pull/23586) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/pull/9022) 等框架官方支持，欢迎从官方入口直接使用！更多框架如 [Ollama](https://github.com/ollama/ollama/pull/12078) 和 [SGLang](https://github.com/sgl-project/sglang/pull/9610) 的官方支持正在持续适配中！

* [2025.08.26] 🔥🔥🔥 我们开源了 MiniCPM-V 4.5，其视觉性能超越了 GPT-4o-latest、Gemini-2.0 Pro 和 Qwen2.5-VL 72B。它不仅延续并强化了 MiniCPM-V 的热门能力，还带来了诸多实用的新功能。欢迎试用！

* [2025.08.02] 🚀🚀🚀 我们开源了 MiniCPM-V 4.0，该模型在图像理解能力上超越了 GPT-4.1-mini-20250414。该模型不仅继承了 MiniCPM-V 2.6 的众多实用特性，还大幅提升了推理效率。我们还同步开源了适用于 iPhone 和 iPad 的 iOS 应用，欢迎试用！

* [2025.08.01] ⭐️⭐️⭐️ 我们开源了 [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)，提供针对不同人群的全场景使用指南，配合最新的[文档网站](https://minicpm-o.readthedocs.io/en/latest/index.html)上手更轻松！

* [2025.06.20] ⭐️⭐️⭐️ MiniCPM-o 的 Ollama [官方仓库](https://ollama.com/openbmb)正式支持 MiniCPM-o 2.6 等模型啦，欢迎[一键使用](https://ollama.com/openbmb/minicpm-o2.6)！

* [2025.03.01] 🚀🚀🚀 MiniCPM-o 系列的对齐技术 RLAIF-V 被 CVPR 2025 接收为 Highlights 了！其[代码](https://github.com/RLHF-V/RLAIF-V)、[数据](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)、[论文](https://arxiv.org/abs/2405.17220)均已开源。

* [2025.01.24] 📢📢📢 MiniCPM-o 2.6 技术报告已发布! 欢迎点击[这里](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9)查看.

* [2025.01.23] 💡💡💡 MiniCPM-o 2.6 现在已被北大团队开发的 [Align-Anything](https://github.com/PKU-Alignment/align-anything)，一个用于对齐全模态大模型的框架集成，支持 DPO 和 SFT 在视觉和音频模态上的微调。欢迎试用！

* [2025.01.19] ⭐️⭐️⭐️ MiniCPM-o 在 GitHub Trending 上登顶， Hugging Face Trending 上也达到了第二！

* [2025.01.19] 📢 **注意!** 我们正在努力将 MiniCPM-o 2.6 的支持合并到 llama.cpp、Ollama、vLLM 的官方仓库，但还未完成。请大家暂时先使用我们提供的 fork 来进行部署：[llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md)、[Ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md)、[vllm](./minicpm_o2dot6_zh.md)。 **合并完成前，使用官方仓库可能会导致不可预期的问题**。

* [2025.01.17] 我们更新了 MiniCPM-o 2.6 int4 量化版本的使用方式，解决了模型初始化的问题，欢迎点击[这里](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4)试用！

* [2025.01.13] 🔥🔥🔥 我们开源了 MiniCPM-o 2.6，该模型视觉、语音和多模态流式能力达到了 GPT-4o-202405 级别，进一步优化了 MiniCPM-V 2.6 的众多亮点能力，还支持了很多有趣的新功能。欢迎试用！

## 2024

* [2024.08.17] 🚀🚀🚀 llama.cpp [官方仓库](https://github.com/ggerganov/llama.cpp)正式支持 MiniCPM-V 2.6 啦！点击[这里](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf)查看各种大小的 GGUF 版本。

* [2024.08.15] MiniCPM-V 2.6 现在支持多图像 SFT。有关更多详细信息，请参阅[微调文档](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune)

* [2024.08.14] MiniCPM-V 2.6 现在可以通过 SWIFT 框架 [微调](https://github.com/modelscope/ms-swift/issues/1613) 了！

* [2024.08.10] 🚀🚀🚀 llama.cpp [官方仓库](https://github.com/ggerganov/llama.cpp)正式支持 MiniCPM-Llama3-V 2.5 啦！点击[这里](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/tree/main)查看各种大小的 GGUF 版本。

* [2024.08.06] 🔥🔥🔥 我们开源了 MiniCPM-V 2.6，该模型在单图、多图和视频理解方面取得了优于 GPT-4V 的表现。我们还进一步提升了 MiniCPM-Llama3-V 2.5 的多项亮点能力，并首次支持了 iPad 上的实时视频理解。欢迎试用！

* [2024.08.03] MiniCPM-Llama3-V 2.5 技术报告已发布！欢迎点击[这里](https://arxiv.org/abs/2408.01800)查看。

* [2024.07.19] MiniCPM-Llama3-V 2.5 现已支持[vLLM](./minicpm_llama3_v2dot5.md) ！

* [2024.06.03] 现在，你可以利用多张低显存显卡（12G/16G）进行GPU串行推理。详情请参见该[文档](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md)配置。

* [2024.05.28] 💫 我们现在支持 MiniCPM-Llama3-V 2.5 的 LoRA 微调，更多内存使用统计信息可以在[这里](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune#model-fine-tuning-memory-usage-statistics)找到。

* [2024.05.28] 💥 MiniCPM-Llama3-V 2.5 现在在 llama.cpp 和 Ollama 中完全支持其功能！**请拉取我们最新的 fork 来使用**：[llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md) & [ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5)。我们还发布了各种大小的 GGUF 版本，请点击[这里](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/tree/main)查看。请注意，**目前官方仓库尚未支持 MiniCPM-Llama3-V 2.5**，我们也正积极推进将这些功能合并到 llama.cpp & ollama 官方仓库，敬请关注！

* [2024.05.25] MiniCPM-Llama3-V 2.5 [支持流式输出和自定义系统提示词](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5#usage)了，欢迎试用!

* [2024.05.24] 我们开源了 MiniCPM-Llama3-V 2.5 [gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf)，支持 [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md) 推理！实现端侧 6-8 tokens/s 的流畅解码，欢迎试用！

* [2024.05.23] 🔥🔥🔥 MiniCPM-V 在 GitHub Trending 和 Hugging Face Trending 上登顶！MiniCPM-Llama3-V 2.5 Demo 被 Hugging Face 的 Gradio 官方账户推荐，欢迎点击[这里](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5)体验！

* [2024.05.23] 🔍 我们添加了Phi-3-vision-128k-instruct 与 MiniCPM-Llama3-V 2.5的全面对比，包括基准测试评估、多语言能力和推理效率 🌟📊🌍🚀。点击[这里](./compare_with_phi-3_vision.md)查看详细信息。

* [2024.05.20] 我们开源了 MiniCPM-Llama3-V 2.5，增强了 OCR 能力，支持 30 多种语言，并首次在端侧实现了 GPT-4V 级的多模态能力！我们提供了[高效推理](./minicpm_llama3_v2dot5.md)和[简易微调](../finetune/readme.md)的支持，欢迎试用！

* [2024.04.23] 我们增加了MiniCPM-V 2.0对 [vLLM](./minicpm_v2.md) 的支持，欢迎体验！

* [2024.04.18] 我们在 HuggingFace Space 新增了 MiniCPM-V 2.0 的 [demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)，欢迎体验！

* [2024.04.17] MiniCPM-V 2.0 现在支持用户部署本地 [WebUI Demo](../web_demos/web_demo.py) 了，欢迎试用!

* [2024.04.15] MiniCPM-V 2.0 现在可以通过 SWIFT 框架 [微调](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md) 了，支持流式输出!

* [2024.04.12] 我们开源了 MiniCPM-V 2.0，该模型刷新了 OCRBench 开源模型最佳成绩，在场景文字识别能力上比肩 Gemini Pro，同时还在综合了 11 个主流多模态大模型评测基准的 <a href="https://rank.opencompass.org.cn/leaderboard-multimodal">OpenCompass</a> 榜单上超过了 Qwen-VL-Chat 9.6B、CogVLM-Chat 17B 和 Yi-VL 34B 等更大参数规模的模型！点击 <a href="https://openbmb.vercel.app/minicpm-v-2">这里</a> 查看 MiniCPM-V 2.0 技术博客。

* [2024.03.14] MiniCPM-V 现在支持 SWIFT 框架下的[微调](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md)了，感谢 [Jintao](https://github.com/Jintao-Huang) 的贡献！

* [2024.03.01] MiniCPM-V 现在支持在 Mac 电脑上进行部署！

* [2024.02.01] 我们开源了 MiniCPM-V 和 OmniLMM-12B，分别可以支持高效的端侧部署和同规模领先的多模态能力！
