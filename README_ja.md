<div align="center">

<img src="./assets/minicpmv.png" width="300em" ></img>

**あなたのスマートフォンでシングルイメージ、マルチイメージ、ビデオに対応するGPT-4VレベルのMLLM**

  <strong>[中文](./README_zh.md) |
  [English](./README_en.md) |
  日本語</strong>

<a href="docs/wechat.md" target="_blank"> 💬 WeChat</a> に参加してください


<p align="center">
  MiniCPM-V 2.6 <a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">🤗</a> <a href="https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6">🤖</a> | MiniCPM-Llama3-V 2.5  <a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/">🤗</a> <a href="https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5">🤖</a> |
  <a href=https://arxiv.org/abs/2408.01800>MiniCPM-Llama3-V 2.5 技術報告</a> 
</p>

</div>


**MiniCPM-V**は、視覚と言語の理解を目的としたエンドサイドのマルチモーダルLLM（MLLM）シリーズです。これらのモデルは画像、ビデオ、テキストを入力として受け取り、高品質なテキスト出力を提供します。2024年2月以降、私たちはこのモデルの5つのバージョンをリリースし、**高い性能と効率的なデプロイメント**を目指しています。このシリーズで現在最も注目すべきモデルには以下が含まれます：

- **MiniCPM-V 2.6**: 🔥🔥🔥 MiniCPM-Vシリーズの最新かつ最も強力なモデルです。合計8Bのパラメータを持ち、**シングルイメージ、マルチイメージ、ビデオの理解においてGPT-4Vを超えます**。**GPT-4o mini、Gemini 1.5 Pro、Claude 3.5 Sonnet**をシングルイメージの理解で上回り、MiniCPM-Llama3-V 2.5の強力なOCR機能、信頼性のある動作、多言語対応、エンドサイドデプロイメントなどの機能をさらに進化させました。その優れたトークン密度により、MiniCPM-V 2.6は初めてiPadなどのエンドサイドデバイスでリアルタイムのビデオ理解をサポートできます。

- **MiniCPM-V 2.0**: MiniCPM-Vシリーズの最軽量モデルです。2Bのパラメータを持ち、Yi-VL 34B、CogVLM-Chat 17B、Qwen-VL-Chat 10Bなどの大規模モデルを総合性能で上回ります。任意のアスペクト比と最大1.8百万ピクセル（例：1344x1344）の画像入力を受け入れることができ、シーンテキストの理解でGemini Proと同等の性能を達成し、低い幻覚率でGPT-4Vに匹敵します。


## ニュース <!-- omit in toc -->

#### 📌 ピン留め
* [2024.08.06] 🔥🔥🔥 MiniCPM-V 2.6をオープンソース化しました。これはシングルイメージ、マルチイメージ、ビデオの理解でGPT-4Vを上回ります。MiniCPM-Llama3-V 2.5の人気機能を進化させ、iPadでのリアルタイムビデオ理解をサポートします。今すぐお試しください！
* [2024.08.03] MiniCPM-Llama3-V 2.5の技術報告がリリースされました！詳細は[こちら](https://arxiv.org/abs/2408.01800)をご覧ください。
* [2024.07.19] MiniCPM-Llama3-V 2.5がvLLMをサポートしました！詳細は[こちら](#inference-with-vllm)をご覧ください。
* [2024.05.28] 🚀🚀🚀 MiniCPM-Llama3-V 2.5はllama.cppとollamaで完全にサポートされました！最新のコードを**私たちの提供するフォーク**からプルしてください（[llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md)、[ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5)）。さまざまなサイズのGGUFモデルが[こちら](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/tree/main)で利用可能です。MiniCPM-Llama3-V 2.5シリーズは**まだ公式リポジトリではサポートされていません**が、PRのマージに向けて努力しています。続報をお待ちください！
* [2024.05.28] 💫 MiniCPM-Llama3-V 2.5のLoRAファインチューニングをサポートしました。2つのV100 GPUのみを使用します！詳細な統計情報は[こちら](https://github.com/OpenBMB/MiniCPM-V/tree/main/finetune#model-fine-tuning-memory-usage-statistics)をご覧ください。
* [2024.05.23] 🔍 Phi-3-vision-128k-instructとMiniCPM-Llama3-V 2.5の包括的な比較をリリースしました。ベンチマーク評価、多言語対応、推論効率などを含みます🌟📊🌍🚀。詳細は[こちら](./docs/compare_with_phi-3_vision.md)をご覧ください。
* [2024.05.23] 🔥🔥🔥 MiniCPM-VがGitHub TrendingとHugging Face Trendingでトップに立ちました！Hugging Face Gradioの公式アカウントに推薦されたデモは[こちら](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5)で利用可能です。ぜひお試しください！

<br>

<details>
<summary>クリックして詳細なニュースを表示</summary>

* [2024.06.03] MiniCPM-Llama3-V 2.5を複数の低VRAM GPU（12 GBまたは16 GB）で実行できます。詳細は[こちら](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md)をご覧ください。
* [2024.05.25] MiniCPM-Llama3-V 2.5はストリーミング出力とカスタマイズされたシステムプロンプトをサポートしました。お試しください[こちら](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5#usage)！
* [2024.05.24] MiniCPM-Llama3-V 2.5 [gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf)をリリースしました。これは[llama.cpp](#inference-with-llamacpp)推論をサポートし、モバイルフォンで6〜8トークン/秒のスムーズなデコードを提供します。今すぐお試しください！
* [2024.05.20] MiniCPM-Llama3-V 2.5をオープンソース化しました。OCR機能が向上し、30以上の言語をサポートします。これはエンドサイドでGPT-4Vレベルの性能を達成した最初のMLLMです！[効率的な推論](#deployment-on-mobile-phone)と[簡単なファインチューニング](./finetune/readme.md)を提供します。今すぐお試しください！
* [2024.04.23] MiniCPM-V-2.0がvLLMをサポートしました！詳細は[こちら](#inference-with-vllm)をご覧ください。
* [2024.04.18] MiniCPM-V 2.0のデモをホストするためにHuggingFace Spaceを作成しました。[こちら](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)でご覧ください。
* [2024.04.17] MiniCPM-V-2.0が[WebUI Demo](#webui-demo)のデプロイをサポートしました！
* [2024.04.15] MiniCPM-V-2.0がSWIFTフレームワークでの[ファインチューニング](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md)をサポートしました！
* [2024.04.12] MiniCPM-V 2.0をオープンソース化しました。これはシーンテキストの理解でGemini Proと同等の性能を達成し、強力なQwen-VL-Chat 9.6BとYi-VL 34Bを上回ります。<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">OpenCompass</a>での評価結果をご覧ください。MiniCPM-V 2.0の技術ブログは<a href="https://openbmb.vercel.app/minicpm-v-2">こちら</a>です。
* [2024.03.14] MiniCPM-VがSWIFTフレームワークでの[ファインチューニング](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md)をサポートしました。貢献してくれた[Jintao](https://github.com/Jintao-Huang)に感謝します！
* [2024.03.01] MiniCPM-VがMacでのデプロイをサポートしました！
* [2024.02.01] MiniCPM-VとOmniLMM-12Bをオープンソース化しました。これらはそれぞれ効率的なエンドサイドデプロイメントと強力なマルチモーダル機能をサポートします。
</details>


## 目次 <!-- omit in toc -->


- [MiniCPM-V 2.6](#minicpm-v-26)
- [MiniCPM-Llama3-V 2.5](#minicpm-llama3-v-25)
- [MiniCPM-V 2.0](#minicpm-v-20)
- [Gradioデモでチャットする 🤗](#chat-with-our-demo-on-gradio-)
- [インストール](#install)
- [推論](#inference)
  - [モデルズー](#model-zoo)
  - [マルチターン会話](#multi-turn-conversation)
    - [複数の画像でチャット](#chat-with-multiple-images)
    - [インコンテキスト少数ショット学習](#in-context-few-shot-learning)
    - [ビデオでチャット](#chat-with-video)
  - [複数のGPUでの推論](#inference-on-multiple-gpus)
  - [Macでの推論](#inference-on-mac)
  - [モバイルフォンでのデプロイ](#deployment-on-mobile-phone)
  - [llama.cppでの推論](#inference-with-llamacpp)
  - [ollamaでの推論](#inference-with-ollama)
  - [vLLMでの推論](#inference-with-vllm)
- [ファインチューニング](#fine-tuning)
- [FAQs](#faqs)


## MiniCPM-V 2.6

**MiniCPM-V 2.6**は、MiniCPM-Vシリーズの最新かつ最も強力なモデルです。このモデルはSigLip-400MとQwen2-7Bを基に構築され、合計8Bのパラメータを持ちます。MiniCPM-Llama3-V 2.5に比べて大幅な性能向上を示し、マルチイメージとビデオ理解の新機能を導入しています。MiniCPM-V 2.6の主な特徴は以下の通りです：

- 🔥 **先進的な性能。**
  MiniCPM-V 2.6は、8つの人気ベンチマークを包括的に評価するOpenCompassの最新バージョンで平均スコア65.2を達成しました。**わずか8Bのパラメータで、GPT-4o mini、GPT-4V、Gemini 1.5 Pro、Claude 3.5 Sonnetなどの広く使用されている商用モデルをシングルイメージ理解で上回ります**。

- 🖼️ **マルチイメージ理解とインコンテキスト学習。** MiniCPM-V 2.6は**複数の画像に対する会話と推論**も行えます。Mantis-Eval、BLINK、Mathverse mv、Sciverse mvなどの人気のマルチイメージベンチマークで**最先端の性能**を達成し、インコンテキスト学習能力も示しています。

- 🎬 **ビデオ理解。** MiniCPM-V 2.6は**ビデオ入力も受け入れ**、会話を行い、時空間情報の詳細なキャプションを提供します。**GPT-4V、Claude 3.5 Sonnet、LLaVA-NeXT-Video-34B**をビデオMMEで字幕あり/なしの両方で上回ります。

- 💪 **強力なOCR機能とその他の機能。**
  MiniCPM-V 2.6は任意のアスペクト比と最大1.8百万ピクセル（例：1344x1344）の画像を処理できます。**OCRBenchで最先端の性能を達成し、GPT-4o、GPT-4V、Gemini 1.5 Proなどの商用モデルを上回ります**。
  最新の[RLAIF-V](https://github.com/RLHF-V/RLAIF-V/)と[VisCPM](https://github.com/OpenBMB/VisCPM)技術に基づき、**信頼性のある動作**を特徴とし、Object HalBenchでGPT-4oやGPT-4Vよりもはるかに低い幻覚率を示し、英語、中国語、ドイツ語、フランス語、イタリア語、韓国語などの**多言語対応**をサポートします。

- 🚀 **優れた効率性。**
  そのフレンドリーなサイズに加えて、MiniCPM-V 2.6は**最先端のトークン密度**（つまり、各視覚トークンにエンコードされるピクセル数）も示しています。**1.8Mピクセルの画像を処理する際に640トークンしか生成せず、ほとんどのモデルよりも75％少ない**。これにより、推論速度、最初のトークンの遅延、メモリ使用量、消費電力が直接向上します。その結果、MiniCPM-V 2.6はiPadなどのエンドサイドデバイスで**リアルタイムのビデオ理解**を効率的にサポートできます。

-  💫  **簡単な使用。**
MiniCPM-V 2.6はさまざまな方法で簡単に使用できます：（1）[llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpmv-main/examples/llava/README-minicpmv2.6.md)と[ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md)のサポートにより、ローカルデバイスで効率的なCPU推論が可能、（2）[int4](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4)と[GGUF](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf)形式の量子化モデルが16サイズで提供、（3）[vLLM](#inference-with-vllm)のサポートにより、高スループットとメモリ効率の高い推論が可能、（4）新しいドメインやタスクでのファインチューニング、（5）[Gradio](#chat-with-our-demo-on-gradio)を使用してローカルWebUIデモを迅速に設定、（6）オンラインWeb[デモ](https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6)。

### 評価  <!-- omit in toc -->
<div align="center">
    <img src=assets/radar_final.png width=66% />
</div>

<details>
<summary>OpenCompass, MME, MMVet, OCRBench, MMMU, MathVista, MMB, AI2D, TextVQA, DocVQA, HallusionBench, Object HalBenchのシングルイメージ結果を表示</summary>
<div align="center">

<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">モデル</th>
            <th>サイズ</th>
            <th>トークン密度<sup>+</sup></th>
            <th>OpenCompass</th>
            <th>MME</th>
            <th>MMVet</th>
            <th>OCRBench</th>
            <th>MMMU val</th>
            <th>MathVista mini</th>
            <th>MMB1.1 test</th>
            <th>AI2D</th>
            <th>TextVQA val</th>
            <th>DocVQA test</th>
            <th>HallusionBench</th>
            <th>Object HalBench</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="15" align="left"><strong>商用</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o</td>
            <td>-</td>
            <td>1088</td>
            <td>69.9</td>
            <td>2328.7</td>
            <td>69.1</td>
            <td>736</td>
            <td>69.2</td>
            <td>61.3</td>
            <td>82.2</td>
            <td>84.6</td>
            <td>-</td>
            <td>92.8</td>
            <td>55.0</td>
            <td>17.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Claude 3.5 Sonnet</td>
            <td>-</td>
            <td>750</td>
            <td>67.9</td>
            <td>1920.0</td>
            <td>66.0</td>
            <td>788</td>
            <td>65.9</td>
            <td>61.6</td>
            <td>78.5</td>
            <td>80.2</td>
            <td>-</td>
            <td>95.2</td>
            <td>49.9</td>
            <td>13.8</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini 1.5 Pro</td>
            <td>-</td>
            <td>-</td>
            <td>64.4</td>
            <td>2110.6</td>
            <td>64.0</td>
            <td>754</td>
            <td>60.6</td>
            <td>57.7</td>
            <td>73.9</td>
            <td>79.1</td>
            <td>73.5</td>
            <td>86.5</td>
            <td>45.6</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4o mini</td>
            <td>-</td>
            <td>1088</td>
            <td>64.1</td>
            <td>2003.4</td>
            <td>66.9</td>
            <td>785</td>
            <td>60.0</td>
            <td>52.4</td>
            <td>76.0</td>
            <td>77.8</td>
            <td>-</td>
            <td>-</td>
            <td>46.1</td>
            <td>12.4</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V</td>
            <td>-</td>
            <td>1088</td>
            <td>63.5</td>
            <td>2070.2</td>
            <td>67.5</td>
            <td>656</td>
            <td>61.7</td>
            <td>54.7</td>
            <td>79.8</td>
            <td>78.6</td>
            <td>78.0</td>
            <td>87.2</td>
            <td>43.9</td>
            <td>14.2</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Step-1V</td>
            <td>-</td>
            <td>-</td>
            <td>59.5</td>
            <td>2206.4</td>
            <td>63.3</td>
            <td>625</td>
            <td>49.9</td>
            <td>44.8</td>
            <td>78.0</td>
            <td>79.2</td>
            <td>71.6</td>
            <td>-</td>
            <td>48.4</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen-VL-Max</td>
            <td>-</td>
            <td>784</td>
            <td>58.3</td>
            <td>2281.7</td>
            <td>61.8</td>
            <td>684</td>
            <td>52.0</td>
            <td>43.4</td>
            <td>74.6</td>
            <td>75.7</td>
            <td>79.5</td>
            <td>93.1</td>
            <td>41.2</td>
            <td>13.4</td>
        </tr>
        <tr>
            <td colspan="15" align="left"><strong>オープンソース</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Yi-34B</td>
            <td>34B</td>
            <td>157</td>
            <td>55.0</td>
            <td>2006.5</td>
            <td>50.7</td>
            <td>574</td>
            <td>48.8</td>
            <td>40.4</td>
            <td>77.8</td>
            <td>78.9</td>
            <td>69.3</td>
            <td>-</td>
            <td>34.8</td>
            <td>12.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Mini-Gemini-HD-34B</td>
            <td>34B</td>
            <td>157</td>
            <td>-</td>
            <td>2141.0</td>
            <td>59.3</td>
            <td>518</td>
            <td>48.0</td>
            <td>43.3</td>
            <td>-</td>
            <td>80.5</td>
            <td>74.1</td>
            <td>78.9</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Cambrian-34B</td>
            <td>34B</td>
            <td>1820</td>
            <td>58.3</td>
            <td>2049.9</td>
            <td>53.2</td>
            <td>591</td>
            <td>50.4</td>
            <td>50.3</td>
            <td>77.8</td>
            <td>79.5</td>
            <td>76.7</td>
            <td>75.5</td>
            <td>41.6</td>
            <td>14.7</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GLM-4V-9B</td>
            <td>13B</td>
            <td>784</td>
            <td>59.1</td>
            <td>2018.8</td>
            <td>58.0</td>
            <td>776</td>
            <td>46.9</td>
            <td>51.1</td>
            <td>67.9</td>
            <td>71.2</td>
            <td>-</td>
            <td>-</td>
            <td>45.0</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2-8B</td>
            <td>8B</td>
            <td>706</td>
            <td>64.1</td>
            <td>2215.1</td>
            <td>54.3</td>
            <td>794</td>
            <td><strong>51.2</strong></td>
            <td>58.3</td>
            <td><strong>79.4</strong></td>
            <td><strong>83.6</strong></td>
            <td>77.4</td>
            <td><strong>91.6</strong></td>
            <td>45.0</td>
            <td>21.3</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">MiniCPM-Llama-V 2.5</td>
            <td>8B</td>
            <td>1882</td>
            <td>58.8</td>
            <td>2024.6</td>
            <td>52.8</td>
            <td>725</td>
            <td>45.8</td>
            <td>54.3</td>
            <td>72.0</td>
            <td>78.4</td>
            <td>76.6</td>
            <td>84.8</td>
            <td>42.4</td>
            <td>10.3</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>2822</strong></td>
            <td><strong>65.2</strong></td>
            <td><strong>2348.4</strong>*</td>
            <td><strong>60.0</strong></td>
            <td><strong>852</strong>*</td>
            <td>49.8*</td>
            <td><strong>60.6</strong></td>
            <td>78.0</td>
            <td>82.1</td>
            <td><strong>80.1<strong></td>
            <td>90.8</td>
            <td><strong>48.1</strong>*</td>
            <td><strong>8.2</strong></td>
        </tr>
    </tbody>
</table>

</div>
* このベンチマークは思考の連鎖プロンプティングを使用して評価しました。具体的には、MMEではこの技術をCognitionセットにのみ使用しました。

<sup>+</sup> トークン密度：最大解像度で各視覚トークンにエンコードされるピクセル数、つまり最大解像度でのピクセル数/視覚トークン数。

注：商用モデルについては、公式APIドキュメントで定義された画像エンコード課金戦略に基づいてトークン密度を計算し、上限を推定しています。

</details>


<details>
<summary>Mantis Eval, BLINK, Mathverse mv, Sciverse mv, MIRBのマルチイメージ結果を表示</summary>
<div align="center">

<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">モデル</th>
            <th>サイズ</th>
            <th>Mantis Eval</th>
            <th>BLINK val</th>
            <th>Mathverse mv</th>
            <th>Sciverse mv</th>
            <th>MIRB</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="7" align="left"><strong>商用</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V</td>
            <td>-</td>
            <td>62.7</td>
            <td>54.6</td>
            <td>60.3</td>
            <td>66.9</td>
            <td>53.1</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Interleave-14B</td>
            <td>14B</td>
            <td>66.4</td>
            <td>52.6</td>
            <td>32.7</td>
            <td>30.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="7" align="left"><strong>オープンソース</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Emu2-Chat</td>
            <td>37B</td>
            <td>37.8</td>
            <td>36.2</td>
            <td>-</td>
            <td>27.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM</td>
            <td>17B</td>
            <td>45.2</td>
            <td>41.1</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">VPG-C</td>
            <td>7B</td>
            <td>52.4</td>
            <td>43.1</td>
            <td>24.3</td>
            <td>23.1</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">VILA 8B</td>
            <td>8B</td>
            <td>51.2</td>
            <td>39.3</td>
            <td>-</td>
            <td>36.5</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternLM-XComposer-2.5</td>
            <td>8B</td>
            <td>53.1*</td>
            <td>48.9</td>
            <td>32.1*</td>
            <td>-</td>
            <td>42.5</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2-8B</td>
            <td>8B</td>
            <td>59.0*</td>
            <td>50.9</td>
            <td>30.5*</td>
            <td>34.4*</td>
            <td><strong>56.9*</strong></td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>69.1</strong></td>
            <td><strong>53.0</strong></td>
            <td><strong>84.9</strong></td>
            <td><strong>74.9</strong></td>
            <td>53.8</td>
        </tr>
    </tbody>
</table>

</div>
* 正式にリリースされたチェックポイントを自分で評価しました。
</details>

<details>
<summary>Video-MMEとVideo-ChatGPTのビデオ結果を表示</summary>
<div align="center">
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">モデル</th>
            <th>サイズ</th>
            <th colspan="2">Video-MME</th>
            <th colspan="5">Video-ChatGPT</th>
        </tr>
        <tr>
            <th align="left"></th>
            <th></th>
            <th>字幕なし</th>
            <th>字幕あり</th>
            <th>正確性</th>
            <th>詳細</th>
            <th>コンテキスト</th>
            <th>時系列</th>
            <th>一貫性</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="9" align="left"><strong>商用</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Claude 3.5 Sonnet</td>
            <td>-</td>
            <td>60.0</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V</td>
            <td>-</td>
            <td>59.9</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="9" align="left"><strong>オープンソース</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-7B</td>
            <td>7B</td>
            <td>-</td>
            <td>-</td>
            <td>3.39</td>
            <td>3.29</td>
            <td>3.92</td>
            <td>2.60</td>
            <td>3.12</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-34B</td>
            <td>34B</td>
            <td>-</td>
            <td>-</td>
            <td>3.29</td>
            <td>3.23</td>
            <td>3.83</td>
            <td>2.51</td>
            <td>3.47</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM2-Video</td>
            <td>12B</td>
            <td>-</td>
            <td>-</td>
            <td>3.49</td>
            <td><strong>3.46</strong></td>
            <td>3.23</td>
            <td><strong>2.98</strong></td>
            <td><strong>3.64</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LongVA</td>
            <td>7B</td>
            <td>52.4</td>
            <td>54.3</td>
            <td>3.05</td>
            <td>3.09</td>
            <td>3.77</td>
            <td>2.44</td>
            <td><strong>3.64</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternVL2-8B</td>
            <td>8B</td>
            <td>54.0</td>
            <td>56.9</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">InternLM-XComposer-2.5</td>
            <td>8B</td>
            <td>55.8</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT-Video</td>
            <td>32B</td>
            <td>60.2</td>
            <td>63.0</td>
            <td>3.48</td>
            <td>3.37</td>
            <td><strong>3.95</strong></td>
            <td>2.64</td>
            <td>3.28</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.6</td>
            <td>8B</td>
            <td><strong>60.9</strong></td>
            <td><strong>63.6</strong></td>
            <td><strong>3.59</strong></td>
            <td>3.28</td>
            <td>3.93</td>
            <td>2.73</td>
            <td>3.62</td>
        </tr>
    </tbody>
</table>
</div>
</details>


<details>
<summary>TextVQA, VizWiz, VQAv2, OK-VQAの少数ショット結果を表示</summary>
<div align="center">
<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">モデル</th>
            <th>サイズ</th>
            <th>ショット</th>
            <th>TextVQA val</th>
            <th>VizWiz test-dev</th>
            <th>VQAv2 test-dev</th>
            <th>OK-VQA val</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">Flamingo</td>
            <td rowspan="3">80B</td>
            <td>0*</td>
            <td>35.0</td>
            <td>31.6</td>
            <td>56.3</td>
            <td>40.6</td>
        </tr>
        <tr>
            <td>4</td>
            <td>36.5</td>
            <td>39.6</td>
            <td>63.1</td>
            <td><strong>57.4</strong></td>
        </tr>
        <tr>
            <td>8</td>
            <td>37.3</td>
            <td>44.8</td>
            <td>65.6</td>
            <td>57.5</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">IDEFICS</td>
            <td rowspan="3">80B</td>
            <td>0*</td>
            <td>30.9</td>
            <td>36.0</td>
            <td>60.0</td>
            <td>45.2</td>
        </tr>
        <tr>
            <td>4</td>
            <td>34.3</td>
            <td>40.4</td>
            <td>63.6</td>
            <td>52.4</td>
        </tr>
        <tr>
            <td>8</td>
            <td>35.7</td>
            <td>46.1</td>
            <td>64.8</td>
            <td>55.1</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">OmniCorpus</td>
            <td rowspan="3">7B</td>
            <td>0*</td>
            <td>43.0</td>
            <td>49.8</td>
            <td>63.2</td>
            <td>45.5</td>
        </tr>
        <tr>
            <td>4</td>
            <td>45.4</td>
            <td>51.3</td>
            <td>64.5</td>
            <td>46.5</td>
        </tr>
        <tr>
            <td>8</td>
            <td>45.6</td>
            <td>52.2</td>
            <td>64.7</td>
            <td>46.6</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="3">Emu2</td>
            <td rowspan="3">37B</td>
            <td>0</td>
            <td>26.4</td>
            <td>40.4</td>
            <td>33.5</td>
            <td>26.7</td>
        </tr>
        <tr>
            <td>4</td>
            <td>48.2</td>
            <td>54.6</td>
            <td>67.0</td>
            <td>53.2</td>
        </tr>
        <tr>
            <td>8</td>
            <td>49.3</td>
            <td>54.7</td>
            <td>67.8</td>
            <td>54.1</td>
        </tr>
        <tr>
            <td align="left" nowrap="nowrap" rowspan="2">MM1</td>
            <td rowspan="2">30B</td>
            <td>0</td>
            <td>26.2</td>
            <td>40.4</td>
            <td>48.9</td>
            <td>26.7</td>
        </tr>
        <tr>
            <td>8</td>
            <td>49.3</td>
            <td>54.7</td>
            <td><strong>70.9</strong></td>
            <td>54.1</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td align="left" nowrap="nowrap" rowspan="3">MiniCPM-V 2.6<sup>+</sup></td>
            <td rowspan="3">8B</td>
            <td>0</td>
            <td>43.9</td>
            <td>33.8</td>
            <td>45.4</td>
            <td>23.9</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td>4</td>
            <td>63.6</td>
            <td>60.5</td>
            <td>65.5</td>
            <td>50.1</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td>8</td>
            <td><strong>64.6</strong></td>
            <td><strong>63.4</strong></td>
            <td>68.2</td>
            <td>51.4</td>
        </tr>
    </tbody>
</table>


</div>
* はイメージショットがゼロで、フラミンゴの後にテキストショットが2つ追加されたことを示します。

<sup>+</sup> SFTを使用しない事前トレーニングのckptを評価する。
</details>

### 例 <!-- omit in toc -->

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="assets/minicpmv2_6/multi_img-bike.png" alt="Bike" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/multi_img-menu.png" alt="Menu" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/multi_img-code.png" alt="Code" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/ICL-Mem.png" alt="Mem" style="margin-bottom: 5px;">
  <img src="assets/minicpmv2_6/multiling-medal.png" alt="medal" style="margin-bottom: 10px;">
</div>
<details>
  <summary>クリックして他のケースを見る。</summary>
  <div style="display: flex; flex-direction: column; align-items: center;">
    <img src="assets/minicpmv2_6/ICL-elec.png" alt="elec" style="margin-bottom: 5px;">
    <img src="assets/minicpmv2_6/multiling-olympic.png" alt="Menu" style="margin-bottom: 10px;">
  </div>
</details>

私達はMiniCPM-V 2.6をエンドデバイスに導入しています。デモビデオは、iPad Proのエディションなしの生の画面録画です。

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/ai.gif" width=32%/>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="assets/gif_cases/beer.gif" width=32%/>
    </p>
</table>

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/ticket.gif" width=32%/>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="assets/gif_cases/wfh.gif" width=32%/>
    </p>
</table>

<table align="center">
    <p align="center">
      <video src="https://github.com/user-attachments/assets/21f4b818-ede1-4822-920e-91281725c830" width="360" /> </video>
      <!-- <video src="https://github.com/user-attachments/assets/c835f757-206b-4d9c-8e36-70d67b453628" width="360" /> </video> -->
    </p>
</table>

## MiniCPM-Llama3-V 2.5

<details>
<summary>クリックして MiniCPM-Llama3-V 2.5 の詳細を表示</summary>

**MiniCPM-Llama3-V 2.5** は MiniCPM-V シリーズの最新モデルです。このモデルは SigLip-400M と Llama3-8B-Instruct をベースに構築されており、合計 8B のパラメータを備えています。MiniCPM-V 2.0に比べ、パフォーマンスが大幅に向上しています。MiniCPM-Llama3-V 2.5の主な特徴は以下の通りになります:

- 🔥 **一流のパフォーマンス。**
  MiniCPM-Llama3-V 2.5は、11の一般的なベンチマークを総合的に評価するOpenCompassで、平均スコア65.1を達成しました。**わずか 8B のパラメータで、GPT-4V-1106、Gemini Pro、Claude 3、Qwen-VL-Max** のような広く使用されている独自のモデルを凌駕し、他のLlama 3ベースのMLLMを大きく上回ります。

- 💪 **強力なOCR機能。**
  MiniCPM-Llama3-V 2.5は、あらゆるアスペクト比、最大180万画素（例：1344x1344）の画像を処理でき、OCRBenchで **700+ スコアを達成し、GPT-4o、GPT-4V-0409、Qwen-VL-Max、Gemini Pro** などの独自モデルを凌駕しています。最近のユーザーからのフィードバックに基づき、MiniCPM-Llama3-V 2.5では、全文OCR抽出、表からマークダウンへの変換、その他の高ユーティリティ機能が強化され、さらに指示追従能力と複雑な推論能力が強化され、マルチモーダルなインタラクション体験が向上しました。

- 🏆 **信頼できる行動。**
  最新の[RLAIF-V](https://github.com/RLHF-V/RLAIF-V/)方式（[RLHF-V](https://github.com/RLHF-V)[CVPR'24]シリーズの最新技術）を活用したMiniCPM-Llama3-V 2.5は、より信頼性の高い挙動を示します。これは、Object HalBenchで **10.3%** のハルシネーション率を達成し、GPT-4V-1106（13.6%）より低く、オープンソースコミュニティ内で最高レベルの性能を達成しました。[データ公開](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)。

- 🌏 **多言語対応。**
  Llama3の強力な多言語機能と[VisCPM](https://github.com/OpenBMB/VisCPM)のクロスリンガル汎化技術のおかげで、MiniCPM-Llama3-V 2.5は、そのバイリンガル（中国語-英語）マルチモーダル機能を、ドイツ語、フランス語、スペイン語、イタリア語、韓国語などを含む **30 以上の言語に拡張します** [すべてのサポート言語](./assets/minicpm-llama-v-2-5_languages.md)。

- 🚀 **効率的なデプロイ。**
  MiniCPM-Llama3-V 2.5は、**モデルの量子化、CPUの最適化、NPUの最適化、コンパイルの最適化**を体系的に採用し、エンドサイド機器への高効率な導入を実現しています。Qualcomm のチップを搭載した携帯電話向けに、NPUアクセラレーションフレームワークQNNをllama.cppに初めて統合しました。システマティックな最適化の後、MiniCPM-Llama3-V 2.5は、エンドサイドのMLLM画像エンコーディングにおいて**150倍の高速化**、言語デコーディングにおいて**3倍の高速化**を実現しました。

-  💫  **簡単な使用法。**
MiniCPM-Llama3-V 2.5は様々な方法で簡単に使用できます: (1) [llama.cpp](https://github.com/OpenBMB/llama.cpp/blob/minicpm-v2.5/examples/minicpmv/README.md)と[ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5)によるローカルデバイス上での効率的なCPU推論のサポート、(2) [GGUF](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf)による16サイズの量子化モデルのフォーマット、(3) 効率的な[LoRA](https://github. com/OpenBMB/MiniCPM-V/tree/main/finetune#lora-finetuning)による微調整、(4) [ストリーミング出力](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5#usage)、(5) [Gradio](https://github.com/OpenBMB/MiniCPM-V/blob/main/web_demo_2.5.py)と[Streamlit](https://github.com/OpenBMB/MiniCPM-V/blob/main/web_demo_streamlit-2_5.py)による迅速なローカルWebUIデモセットアップ、(6) [HuggingFace Spaces](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5)によるインタラクティブデモ。

### 評価  <!-- omit in toc -->

<div align="center">
    <img src=assets/MiniCPM-Llama3-V-2.5-peformance.png width=66% />
</div>
<details>
<summary>TextVQA、DocVQA、OCRBench、OpenCompass、MME、MMBench、MMMU、MathVista、LLaVA Bench、RealWorld QA、Object HalBench の結果を見るにはクリックしてください。</summary>
<div align="center">

<table style="margin: 0px auto;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>OCRBench</th>
            <th>TextVQA val</th>
            <th>DocVQA test</th>
            <th>Open-Compass</th>
            <th>MME</th>
            <th>MMB test (en)</th>
            <th>MMB test (cn)</th>
            <th>MMMU val</th>
            <th>Math-Vista</th>
            <th>LLaVA Bench</th>
            <th>RealWorld QA</th>
            <th>Object HalBench</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="14" align="left"><strong>商用</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini Pro</td>
            <td>-</td>
            <td>680</td>
            <td>74.6</td>
            <td>88.1</td>
            <td>62.9</td>
            <td>2148.9</td>
            <td>73.6</td>
            <td>74.3</td>
            <td>48.9</td>
            <td>45.8</td>
            <td>79.9</td>
            <td>60.4</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V (2023.11.06)</td>
            <td>-</td>
            <td>645</td>
            <td>78.0</td>
            <td>88.4</td>
            <td>63.5</td>
            <td>1771.5</td>
            <td>77.0</td>
            <td>74.4</td>
            <td>53.8</td>
            <td>47.8</td>
            <td>93.1</td>
            <td>63.0</td>
            <td>86.4</td>
        </tr>
        <tr>
            <td colspan="14" align="left"><strong>オープンソース</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Mini-Gemini</td>
            <td>2.2B</td>
            <td>-</td>
            <td>56.2</td>
            <td>34.2*</td>
            <td>-</td>
            <td>1653.0</td>
            <td>-</td>
            <td>-</td>
            <td>31.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen-VL-Chat</td>
            <td>9.6B</td>
            <td>488</td>
            <td>61.5</td>
            <td>62.6</td>
            <td>51.6</td>
            <td>1860.0</td>
            <td>61.8</td>
            <td>56.3</td>
            <td>37.0</td>
            <td>33.8</td>
            <td>67.7</td>
            <td>49.3</td>
            <td>56.2</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">DeepSeek-VL-7B</td>
            <td>7.3B</td>
            <td>435</td>
            <td>64.7*</td>
            <td>47.0*</td>
            <td>54.6</td>
            <td>1765.4</td>
            <td>73.8</td>
            <td>71.4</td>
            <td>38.3</td>
            <td>36.8</td>
            <td>77.8</td>
            <td>54.2</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Yi-VL-34B</td>
            <td>34B</td>
            <td>290</td>
            <td>43.4*</td>
            <td>16.9*</td>
            <td>52.2</td>
            <td><strong>2050.2</strong></td>
            <td>72.4</td>
            <td>70.7</td>
            <td>45.1</td>
            <td>30.7</td>
            <td>62.3</td>
            <td>54.8</td>
            <td>79.3</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM-Chat</td>
            <td>17.4B</td>
            <td>590</td>
            <td>70.4</td>
            <td>33.3*</td>
            <td>54.2</td>
            <td>1736.6</td>
            <td>65.8</td>
            <td>55.9</td>
            <td>37.3</td>
            <td>34.7</td>
            <td>73.9</td>
            <td>60.3</td>
            <td>73.6</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">TextMonkey</td>
            <td>9.7B</td>
            <td>558</td>
            <td>64.3</td>
            <td>66.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
          <td nowrap="nowrap" align="left">Idefics2</td>
          <td>8.0B</td>
          <td>-</td>
          <td>73.0</td>
          <td>74.0</td>
          <td>57.2</td>
          <td>1847.6</td>
          <td>75.7</td>
          <td>68.6</td>
          <td>45.2</td>
          <td>52.2</td>
          <td>49.1</td>
          <td>60.7</td>
          <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Bunny-LLama-3-8B</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>54.3</td>
            <td>1920.3</td>
            <td>77.0</td>
            <td>73.9</td>
            <td>41.3</td>
            <td>31.5</td>
            <td>61.2</td>
            <td>58.8</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT Llama-3-8B</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>78.2</td>
            <td>-</td>
            <td>1971.5</td>
            <td>-</td>
            <td>-</td>
            <td>41.7</td>
            <td>37.5</td>
            <td>80.1</td>
            <td>60.0</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Phi-3-vision-128k-instruct</td>
            <td>4.2B</td>
            <td>639*</td>
            <td>70.9</td>
            <td>-</td>
            <td>-</td>
            <td>1537.5*</td>
            <td>-</td>
            <td>-</td>
            <td>40.4</td>
            <td>44.5</td>
            <td>64.2*</td>
            <td>58.8*</td>
            <td>-</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 1.0</td>
            <td>2.8B</td>
            <td>366</td>
            <td>60.6</td>
            <td>38.2</td>
            <td>47.5</td>
            <td>1650.2</td>
            <td>64.1</td>
            <td>62.6</td>
            <td>38.3</td>
            <td>28.9</td>
            <td>51.3</td>
            <td>51.2</td>
            <td>78.4</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.0</td>
            <td>2.8B</td>
            <td>605</td>
            <td>74.1</td>
            <td>71.9</td>
            <td>54.5</td>
            <td>1808.6</td>
            <td>69.1</td>
            <td>66.5</td>
            <td>38.2</td>
            <td>38.7</td>
            <td>69.2</td>
            <td>55.8</td>
            <td>85.5</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-Llama3-V 2.5</td>
            <td>8.5B</td>
            <td><strong>725</strong></td>
            <td><strong>76.6</strong></td>
            <td><strong>84.8</strong></td>
            <td><strong>65.1</strong></td>
            <td>2024.6</td>
            <td><strong>77.2</strong></td>
            <td><strong>74.2</strong></td>
            <td><strong>45.8</strong></td>
            <td><strong>54.3</strong></td>
            <td><strong>86.7</strong></td>
            <td><strong>63.5</strong></td>
            <td><strong>89.7</strong></td>
        </tr>
    </tbody>
</table>


</div>
* 公式発表されたチェックポイントを自分たちで評価する。

</details>

<div align="center">
    <img src="assets/llavabench_compare_3.png" width="100%" />
    <br>
    多言語版LLaVA Benchの評価結果
</div>

### Examples <!-- omit in toc -->

<table align="center" >
  <p align="center" >
  <img src="assets/minicpmv-llama3-v2.5/cases_all.png" />
  </p>
</table>

</details>


## MiniCPM-V 2.0

<details>
<summary>MiniCPM-V 2.0 の詳細を見るにはクリックしてください。</summary>


**MiniCPM-V 2.0** は、デプロイに有望な性能を持つ効率的なバージョンである。このモデルはSigLip-400Mと[MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/)をベースに構築されており、perceiver resampler で接続されています。最新バージョンのMiniCPM-V 2.0には、いくつかの特筆すべき特徴があります。

- 🔥 **最先端のパフォーマンス。**

  MiniCPM-V 2.0は、7Bのパラメータを持つモデルの中で、複数のベンチマーク（OCRBench、TextVQA、MME、MMB、MathVistaなどを含む）で**最先端のパフォーマンス**を達成しています。11の一般的なベンチマークを総合的に評価するOpenCompass**では、強力なQwen-VL-Chat 9.6B、CogVLM-Chat 17.4B、Yi-VL 34Bを凌駕しています**。特筆すべきは、MiniCPM-V 2.0は**強力なOCR能力**を示しており、シーンテキスト理解において**Gemini Proに匹敵するパフォーマンス**を達成し、オープンソースモデルの中では**OCRBench**で最先端のパフォーマンス**を発揮している。

- 🏆 **信頼に値する行動。**

  LMMはハルシネーションに悩まされることで知られ、しばしばイメージに基づかないテキストを生成する。MiniCPM-V 2.0は、**マルチモーダルRLHFにより信頼できる振る舞いを実現する**最初のエンドサイドLMMです(最近の[RLHF-V](https://rlhf-v.github.io/) [CVPR'24]シリーズの技術を使用)。これにより、このモデルはObject HalBenchにおいて**ハルシネーション防止においてGPT-4Vに匹敵する**。

- 🌟 **どんなアスペクトでも高解像度の画像を提供。**

  MiniCPM-V 2.0は、**180万画素（例：1344x1344）の画像を任意のアスペクト比で受け入れることができます**。これは、[LLaVA-UHD](https://arxiv.org/pdf/2403.11703.pdf)の最近の技術によって実現されたもので、小さな物体や光学的な文字のような細かい視覚情報のより良い知覚を可能にする。

- ⚡️ **High Efficiency.**

  MiniCPM-V 2.0は、ほとんどのGPUカードやパーソナルコンピュータ**、そして**携帯電話などのエンドデバイス**にも効率的に導入することができます。視覚エンコーディングでは、perceiver resampler によって、画像表現をより少ないトークンに圧縮する。これにより、MiniCPM-V 2.0は、**高解像度画像を扱う場合でも、推論時に有利なメモリコストと速度で動作する**ことができます。

- 🙌 **Bilingual Support.**

  MiniCPM-V 2.0 **英語と中国語の強力なバイリンガル・マルチモーダル機能をサポート**。これは、[VisCPM](https://arxiv.org/abs/2308.12038) [ICLR'24]の技術である、言語間のマルチモーダル能力を一般化することによって可能になる。

### 例 <!-- omit in toc -->

<table align="center">
    <p align="center">
      <img src="assets/minicpmv2-cases_2.png" width=95%/>
    </p>
</table>

我々はMiniCPM-V 2.0をエンドデバイスに導入している。デモビデオはXiaomi 14 Pro（エディションなし）での生の画面録画です。

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/station.gif" width=36%/>
      <img src="assets/gif_cases/london_car.gif" width=36%/>
    </p>
</table>

</details>

## レガシーモデル <!-- omit in toc -->

| モデル                | 導入とガイダンス       |
|:----------------------|:-------------------:|
| MiniCPM-V 1.0  | [ドキュメント](./minicpm_v1.md)   |
| OmniLMM-12B  | [ドキュメント](./omnilmm_en.md)   |


## Gradio🤗 のデモでチャット

私たちは、現在最も人気のあるモデル展開フレームワークであるHugging Face Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> によるオンラインおよびローカルデモを提供しています。ストリーミング出力、プログレスバー、キューイング、アラート、その他の便利な機能をサポートしている。


### オンラインデモ <!-- omit in toc -->

[MiniCPM-V 2.6](https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6) | [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5) | [MiniCPM-V 2.0](https://huggingface.co/spaces/openbmb/MiniCPM-V-2) のオンラインデモをお試しいただくには、こちらをクリックしてください。

### ローカル WebUI デモ <!-- omit in toc -->

以下のコマンドを使えば、Gradio を使った独自のローカル WebUI デモを簡単に構築できる。

```shell
pip install -r requirements.txt
```

```shell
# NVIDIA GPUの場合は、以下を実行:
python web_demo_2.6.py --device cuda

```


## インストール

1. このリポジトリをクローンし、ソースフォルダーに移動

```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
cd MiniCPM-V
```

2. conda 環境の作成

```Shell
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
```

3. 依存関係のインストール

```shell
pip install -r requirements.txt
```

## 推論


### Model Zoo

| モデル           | デバイス | メモリ    | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 説明       | ダウンロード |
|:-----------|:--:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 2.6| GPU | 17 GB  | 最新バージョンは、単一画像、複数画像、ビデオ理解のための最先端のエンドサイド性能を達成。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6) |
| MiniCPM-V 2.6 gguf | CPU | 6 GB  | ggufバージョンは、メモリ使用量が少なく、推論が速い。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6-gguf) |
| MiniCPM-V 2.6 int4 | GPU | 7 GB  | int4量子化バージョンは、GPUメモリ使用量が少ない。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6-int4) |
| MiniCPM-Llama3-V 2.5 | GPU | 19 GB | 強力なエンドサイドのマルチモーダル性能。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5) |
| MiniCPM-Llama3-V 2.5 gguf | CPU  | 6 GB | ggufバージョンは、メモリ使用量が少なく、推論が速い。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf) &nbsp;&nbsp;[<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-gguf) |
| MiniCPM-Llama3-V 2.5 int4 | GPU | 8 GB | int4量子化バージョンは、GPUメモリ使用量が少ない。 |  [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4/) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-int4) |
| MiniCPM-V 2.0 | GPU | 8 GB | 最軽量バージョンは、パフォーマンスと計算コストのバランスを取る。   |  [🤗](https://huggingface.co/openbmb/MiniCPM-V-2) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2) |
| MiniCPM-V 1.0 | GPU | 7 GB | 最軽量バージョンで最速の推論を実現。 |   [🤗](https://huggingface.co/openbmb/MiniCPM-V) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V) |

### マルチターン会話

以下のコードを参考に実行してください。

<div align="center">
<img src="assets/airplane.jpeg" width="500px">
</div>


```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpaまたはflash_attention_2、eager なし
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('./assets/airplane.jpeg').convert('RGB')

# 第1ラウンドチャット
question = "Tell me the model of this aircraft."
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)

# 第2ラウンドチャット
# マルチターン会話のパス履歴コンテキスト
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```

次のような出力が得られます:

```
"The aircraft in the image is an Airbus A380, which can be identified by its large size, double-deck structure, and the distinctive shape of its wings and engines. The A380 is a wide-body aircraft known for being the world's largest passenger airliner, designed for long-haul flights. It has four engines, which are characteristic of large commercial aircraft. The registration number on the aircraft can also provide specific information about the model if looked up in an aviation database."

"The Airbus A380 is a double-deck, wide-body, four-engine jet airliner made by Airbus. It is the world's largest passenger airliner and is known for its long-haul capabilities. The aircraft was developed to improve efficiency and comfort for passengers traveling over long distances. It has two full-length passenger decks, which can accommodate more passengers than a typical single-aisle airplane. The A380 has been operated by airlines such as Lufthansa, Singapore Airlines, and Emirates, among others. It is widely recognized for its unique design and significant impact on the aviation industry."
```

#### 複数の画像を使ったチャット
<details>
<summary> クリックすると、MiniCPM-V 2.6を複数の画像入力で実行するPythonコードが表示されます。 </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpaまたはflash_attention_2、eager なし
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image1 = Image.open('image1.jpg').convert('RGB')
image2 = Image.open('image2.jpg').convert('RGB')
question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'

msgs = [{'role': 'user', 'content': [image1, image2, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```
</details>

#### In-context few-shot learning
<details>
<summary> クリックするとMiniCPM-V 2.6を実行するPythonコードが表示されます。 </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpaまたはflash_attention_2、eager なし
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

question = "production date"
image1 = Image.open('example1.jpg').convert('RGB')
answer1 = "2023.08.04"
image2 = Image.open('example2.jpg').convert('RGB')
answer2 = "2007.04.24"
image_test = Image.open('test.jpg').convert('RGB')

msgs = [
    {'role': 'user', 'content': [image1, question]}, {'role': 'assistant', 'content': [answer1]},
    {'role': 'user', 'content': [image2, question]}, {'role': 'assistant', 'content': [answer2]},
    {'role': 'user', 'content': [image_test, question]}
]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```
</details>

#### ビデオでチャット
<details>
<summary> ビデオ入力でMiniCPM-V 2.6を実行するPythonコードを見るにはクリックしてください。 </summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpaまたはflash_attention_2、eager なし
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES=64 # cuda OOMの場合、より小さい数値を設定する

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path="video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]},
]

# ビデオのデコードパラメータを設定する
params = {}
params["use_image_id"] = False
params["max_slice_nums"] = 2 # cuda OOMおよびビデオ解像度が448*448を超える場合は1を使用する

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)
print(answer)
```
</details>


### マルチGPUでの推論
モデルのレイヤーを複数のGPUに分散することで、複数の低VRAM GPU（12 GBまたは16 GB）でMiniCPM-Llama3-V 2.5を実行できます。複数の低VRAM GPUを使用したモデルのロードと推論の詳細な手順については、こちらの[チュートリアル](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/inference_on_multiple_gpus.md)を参照してください。


### Macでの推論
<details>
<summary>クリックすると、MiniCPM-Llama3-V 2.5をMPS（AppleシリコンまたはAMD GPU）搭載の📨Macで実行する例が表示されます。</summary>

```python
# test.py  16GB以上のメモリが必要。
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, low_cpu_mem_usage=True)
model = model.to(device='mps')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

image = Image.open('./assets/hk_OCR.jpg').convert('RGB')
question = 'Where is this photo taken?'
msgs = [{'role': 'user', 'content': question}]

answer, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True
)
print(answer)
```
コマンドで実行:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py
```
</details>

### モバイルへのデプロイ
MiniCPM-V 2.0 は Android OS 搭載の携帯電話にも導入可能です。🚀 [MiniCPM-V 2.0](https://github.com/OpenBMB/mlc-MiniCPM)をクリックして apk をインストールしてください。

### llama.cppによる推論
MiniCPM-V 2.6はllama.cppで動作するようになりました！詳しくは[our fork of llama.cpp](https://github.com/OpenBMB/llama.cpp/tree/minicpmv-main/examples/llava/README-minicpmv2.6.md)をご覧ください。この実装は、iPad（テスト環境：iPad Pro + M4）で16〜18トークン/秒のスムーズな推論をサポートします。

### ollama の推論
MiniCPM-V 2.6がollamaで動くようになりました！詳しくは [our fork of ollama](https://github.com/OpenBMB/ollama/blob/minicpm-v2.6/examples/minicpm-v2.6/README.md) をご覧ください。この実装は、iPad（テスト環境：iPad Pro + M4）で16〜18トークン/秒のスムーズな推論をサポートします。

### vLLMによる推論

<details>
<summary> vLLMは現在、MiniCPM-V 2.6、MiniCPM-Llama3-V 2.5、MiniCPM-V 2.0を正式にサポートしています。 </summary>

1. Install vLLM(>=0.5.4):
```shell
pip install vllm
```
2. Install timm: (optional, MiniCPM-V 2.0 need timm)
```shell
pip install timm==0.9.10
```
3. 例を実行する(画像用):
```python
from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-2_6"
# 旧モデルにも対応
# MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"
# MODEL_NAME = "HwwwH/MiniCPM-V-2"

image = Image.open("xxx.png").convert("RGB")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=1,
    max_model_len=2048
)

messages = [{
    "role":
    "user",
    "content":
    # 画像数
    "(<image>./</image>)" + \
    "\nWhat is the content of this image?"
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# シングル推論
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
        # マルチイメージの場合、画像の数は `(<image>./</image>)` の数と同じでなければなりません
        # "image": [image, image]
    },
}
# バッチ推論
# inputs = [{
#     "prompt": prompt,
#     "multi_modal_data": {
#         "image": image
#     },
# } for _ in 2]


# 2.6
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
# stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    use_beam_search=True,
    temperature=0,
    best_of=3,
    max_tokens=1024
)

outputs = llm.generate(inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```
4. *video* で使用したい、あるいは `vLLM` の詳細を知りたい場合、[こちら](https://modelbest.feishu.cn/wiki/C2BWw4ZP0iCDy7kkCPCcX2BHnOf?from=from_copylink)をクリックしてください。
</details>

## Fine-tuning

### Simple Fine-tuning <!-- omit in toc -->

Hugging Face for MiniCPM-V 2.0とMiniCPM-Llama3-V 2.5で簡単なファインチューニングをサポートします。

[Reference Document](./finetune/readme.md)

### SWIFT フレームワーク <!-- omit in toc -->

我々は現在、SWIFT フレームワークによる MiniCPM-V シリーズのファインチューニングをサポートしている。SWIFTは、約200のLLMとMLLMのトレーニング、推論、評価、デプロイメントをサポートします。SWIFTは、PEFTが提供する軽量トレーニングソリューションと、NEFTune、LoRA+、LLaMA-PROなどの技術を含む完全なアダプターライブラリをサポートしています。

Best Practices：[MiniCPM-V 1.0](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md), [MiniCPM-V 2.0](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md)

## FAQs
[FAQ](./docs/faqs.md) はこちら

## モデルライセンス <!-- omit in toc -->

* このリポジトリは[Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)ライセンスのもとで公開されています。

* MiniCPM-V モデルウェイトの使用は、[MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md)に厳密に従わなければならない。

* MiniCPMのモデルとウェイトは、学術研究のために完全に無料です。登録のための["アンケート"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g)に記入した後、無料の商用利用も可能です。


## ステートメント <!-- omit in toc -->

LMMであるMiniCPM-Vモデル（OmniLMMを含む）は、大量のマルチモーダルコーパスを学習してコンテンツを生成するが、理解したり、個人的な意見を述べたり、価値判断をしたりすることはできない。MiniCPM-Vモデルによって生成されたものは、モデル開発者の見解や立場を表すものではない

MiniCPM-Vモデルを使用することにより発生する問題（データセキュリティ上の問題、世論のリスク、モデルの誤導、誤用、流布、誤用に起因するリスクや問題を含むがこれらに限定されない）については、当社は一切責任を負いません。


## 機関  <!-- omit in toc -->

このプロジェクトは以下の機関によって開発されています:

- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)
- <img src="assets/zhihu.webp" width="28px"> [Zhihu](https://www.zhihu.com/ )

## 🌟 Star History <!-- omit in toc -->


<table align="center">
    <p align="center">
      <img src="assets/star_history.svg"/>
    </p>
</table>

<!-- <picture>
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
</picture> -->

## 主要技術とその他のマルチモーダルプロジェクト <!-- omit in toc -->

👏 MiniCPM-V の主な技術や、私たちのチームの他のマルチモーダルプロジェクトを探求することを歓迎します。:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)


## 引用 <!-- omit in toc -->

もし私たちのモデル/コード/論文が役に立ったと思われましたら、私たちの論文を引用してください📝 そして私たちに star をつけてください ⭐️！

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```
