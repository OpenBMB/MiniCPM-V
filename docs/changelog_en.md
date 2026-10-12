[← Home](../README.md) · [中文](./changelog_zh.md)

# Changelog

Release announcements and project updates, newest first. Older entries describe support at the time of the announcement; use the current model and deployment guides for current instructions.

## 2026

* [2026.10.12] 🔥🔥🔥 We open-source MiniCPM-V 4.7, the first open-source MoE model to run smoothly on-device without compromising performance, setting a new multimodal performance–efficiency Pareto frontier. Try it now!

* [2026.06.25] 🔥🔥🔥 MiniCPM-V 4.6 has been merged into Ollama’s official model library, bringing ultra-efficient image and video understanding to local Ollama users. Try [it](https://ollama.com/library/minicpm-v4.6) now!

* [2026.05.17] ⭐️⭐️⭐️ We release the [API service](./api.md) of MiniCPM-V 4.6 and MiniCPM-o 4.5.

* [2026.05.11] 🔥🔥🔥 We open-source MiniCPM-V 4.6, with mixed 4x/16x visual token compression. Powered by strong encoding efficiency and its lightweight 1.3B scale, it is our most edge-deployment-friendly model to date, achieving ~1.5x token throughput compared to Qwen3.5 0.8B. Try it now!

* [2026.05.07] 📢📢📢 We release the MiniCPM-o 4.5 technical report, introducing the key techniques behind its real-time full-duplex omni-modal interaction. Read it [here](https://huggingface.co/papers/2604.27393).

* [2026.02.06] 🥳 🥳 🥳 We open-sourced a realtime web demo deployable on your own devices like Mac or GPU. [Try it now](./minicpm_o4dot5_en.md#web-demo-deployment)!

* [2026.02.05] 📢📢📢 We note the web demo may experience latency issues due to network conditions. We are working actively to provide a Docker image for local deployment of the real-time interactive Demo as soon as possible. Please stay tuned!

* [2026.02.03] 🔥🔥🔥 We open-source MiniCPM-o 4.5, which matches Gemini 2.5 Flash on vision and speech, and supports full-duplex multimodal live streaming. Try it now!

## 2025

* [2025.09.18] 📢📢📢 MiniCPM-V 4.5 technical report is now released! See [here](https://arxiv.org/abs/2509.18154).

* [2025.09.01] ⭐️⭐️⭐️ MiniCPM-V 4.5 has been officially supported by [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15575), [vLLM](https://github.com/vllm-project/vllm/pull/23586), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/pull/9022). You are welcome to use it directly through these official channels! Support for additional frameworks such as [Ollama](https://github.com/ollama/ollama/pull/12078) and [SGLang](https://github.com/sgl-project/sglang/pull/9610) is actively in progress.

* [2025.08.26] 🔥🔥🔥 We open-source MiniCPM-V 4.5, which outperforms GPT-4o-latest, Gemini-2.0 Pro, and Qwen2.5-VL 72B. It advances popular capabilities of MiniCPM-V, and brings useful new features. Try it now!

* [2025.08.02] 🚀🚀🚀 We open-source MiniCPM-V 4.0, which outperforms GPT-4.1-mini-20250414 in image understanding. It advances popular features of MiniCPM-V 2.6, and largely improves the efficiency. We also open-source the iOS App on iPhone and iPad. Try it now!

* [2025.08.01] ⭐️⭐️⭐️ We open-sourced the [MiniCPM-V & o Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)! It provides comprehensive guides for diverse user scenarios, paired with our new [Docs Site](https://minicpm-o.readthedocs.io/en/latest/index.html) for smoother onboarding.

* [2025.06.20] ⭐️⭐️⭐️ Our official [Ollama repository](https://ollama.com/openbmb) is released. Try our latest models with [one click](https://ollama.com/openbmb/minicpm-o2.6)！

* [2025.03.01] 🚀🚀🚀 RLAIF-V, the alignment technique of MiniCPM-o, is accepted by CVPR 2025 Highlights！The [code](https://github.com/RLHF-V/RLAIF-V), [dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), [paper](https://arxiv.org/abs/2405.17220) are open-sourced!

* [2025.01.24] 📢📢📢 MiniCPM-o 2.6 technical report is released! See [here](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9).

* [2025.01.23] 💡💡💡 MiniCPM-o 2.6 is now supported by [Align-Anything](https://github.com/PKU-Alignment/align-anything), a framework by PKU-Alignment Team for aligning any-to-any modality large models with human intentions. It supports DPO and SFT fine-tuning on both vision and audio. Try it now!

* [2025.01.19] ⭐️⭐️⭐️ MiniCPM-o tops GitHub Trending and reaches top-2 on Hugging Face Trending!

* [2025.01.19] 📢 **ATTENTION!** We are currently working on merging MiniCPM-o 2.6 into the official repositories of llama.cpp, Ollama, and vllm. Until the merge is complete, please USE OUR LOCAL FORKS of [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md), [Ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md), and [vllm](./minicpm_o2dot6_en.md). **Using the official repositories before the merge may lead to unexpected issues**.

* [2025.01.17] We have updated the usage of MiniCPM-o 2.6 int4 quantization version and resolved the model initialization error. Click [here](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) and try it now!

* [2025.01.13] 🔥🔥🔥 We open-source MiniCPM-o 2.6, which matches GPT-4o-202405 on vision, speech and multimodal live streaming. It advances popular capabilities of MiniCPM-V 2.6, and supports various new fun features. Try it now!

## 2024

* [2024.08.17] 🚀🚀🚀 MiniCPM-V 2.6 is now fully supported by [official](https://github.com/ggerganov/llama.cpp) llama.cpp! GGUF models of various sizes are available [here](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf).

* [2024.08.15] We now also support multi-image SFT. For more details, please refer to the [document](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune).

* [2024.08.14] MiniCPM-V 2.6 now also supports [fine-tuning](https://github.com/modelscope/ms-swift/issues/1613) with the SWIFT framework!

* [2024.08.10] 🚀🚀🚀 MiniCPM-Llama3-V 2.5 is now fully supported by [official](https://github.com/ggerganov/llama.cpp) llama.cpp! GGUF models of various sizes are available [here](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf).

* [2024.08.06] 🔥🔥🔥 We open-source MiniCPM-V 2.6, which outperforms GPT-4V on single image, multi-image and video understanding. It advances popular features of MiniCPM-Llama3-V 2.5, and can support real-time video understanding on iPad. Try it now!

* [2024.08.03] MiniCPM-Llama3-V 2.5 technical report is released! See [here](https://arxiv.org/abs/2408.01800).

* [2024.07.19] MiniCPM-Llama3-V 2.5 supports vLLM now! See [here](./minicpm_llama3_v2dot5.md).

* [2024.06.03] Now, you can run MiniCPM-Llama3-V 2.5 on multiple low VRAM GPUs(12 GB or 16 GB) by distributing the model's layers across multiple GPUs. For more details, check this [link](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md).

* [2024.05.28] 🚀🚀🚀 MiniCPM-Llama3-V 2.5 now fully supports its feature in llama.cpp and Ollama! Please pull the latest code **of our provided forks** ([llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md), [Ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5)). GGUF models in various sizes are available [here](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/tree/main). MiniCPM-Llama3-V 2.5 series is **not supported by the official repositories yet**, and we are working hard to merge PRs. Please stay tuned!

* [2024.05.28] 💫 We now support LoRA fine-tuning for MiniCPM-Llama3-V 2.5, using only 2 V100 GPUs! See more statistics [here](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune#model-fine-tuning-memory-usage-statistics).

* [2024.05.25] MiniCPM-Llama3-V 2.5 now supports streaming outputs and customized system prompts. Try it [here](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5#usage)!

* [2024.05.24] We release the MiniCPM-Llama3-V 2.5 [gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf), which supports [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md) inference and provides a 6~8 token/s smooth decoding on mobile phones. Try it now!

* [2024.05.23] 🔥🔥🔥 MiniCPM-V tops GitHub Trending and Hugging Face Trending! Our demo, recommended by Hugging Face Gradio’s official account, is available [here](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5). Come and try it out!

* [2024.05.23] 🔍 We've released a comprehensive comparison between Phi-3-vision-128k-instruct and MiniCPM-Llama3-V 2.5, including benchmark evaluations, multilingual capabilities, and inference efficiency 🌟📊🌍🚀. Click [here](./compare_with_phi-3_vision.md) to view more details.

* [2024.05.20] We open-soure MiniCPM-Llama3-V 2.5, it has improved OCR capability and supports 30+ languages, representing the first end-side MLLM achieving GPT-4V level performance! We provide [efficient inference](./minicpm_llama3_v2dot5.md) and [simple fine-tuning](../finetune/readme.md). Try it now!

* [2024.04.23] MiniCPM-V-2.0 supports vLLM now! Click [here](./minicpm_v2.md) to view more details.

* [2024.04.18] We create a HuggingFace Space to host the demo of MiniCPM-V 2.0 at [here](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)!

* [2024.04.17] MiniCPM-V-2.0 supports deploying [WebUI Demo](../web_demos/web_demo.py) now!

* [2024.04.15] MiniCPM-V-2.0 now also supports [fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md) with the SWIFT framework, with streaming inference enabled!

* [2024.04.12] We open-source MiniCPM-V 2.0, which achieves comparable performance with Gemini Pro in understanding scene text and outperforms strong Qwen-VL-Chat 9.6B and Yi-VL 34B on <a href="https://rank.opencompass.org.cn/leaderboard-multimodal">OpenCompass</a>, a comprehensive evaluation over 11 popular benchmarks. Click <a href="https://openbmb.vercel.app/minicpm-v-2">here</a> to view the MiniCPM-V 2.0 technical blog.

* [2024.03.14] MiniCPM-V now supports [fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md) with the SWIFT framework. Thanks to [Jintao](https://github.com/Jintao-Huang) for the contribution！

* [2024.03.01] MiniCPM-V can now be deployed on Mac!

* [2024.02.01] We open-source MiniCPM-V and OmniLMM-12B, which support efficient end-side deployment and powerful multimodal capabilities correspondingly.
