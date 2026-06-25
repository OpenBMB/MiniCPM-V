# MiniCPM-V 4.6 Examples Gallery

This gallery contains small, runnable MiniCPM-V 4.6 examples for image, OCR, multi-image, video, low-memory, and local Gradio smoke tests. Each command can optionally write a JSON result that can be assembled into a static HTML gallery.

## Setup

Install the current Transformers path for MiniCPM-V 4.6:

```bash
pip install "transformers[torch]>=5.7.0" torchvision av pillow
```

`av` is a lightweight media decoding option for environments where `torchcodec` has CUDA compatibility issues. If your CUDA and PyTorch versions support `torchcodec`, you can use the installation path from the main README instead.

For the optional local web UI:

```bash
pip install gradio
```

## Image Captioning

Run the default refraction image example:

```bash
python examples/gallery/minicpmv46_inference.py \
  --prompt "Describe this image in one sentence." \
  --output-json outputs/gallery/refraction.json
```

Expected output is a short description of the image, for example:

```text
A glass of water with a red pencil stuck inside it.
```

## OCR

```bash
python examples/gallery/minicpmv46_inference.py \
  --image-url assets/hk_OCR.jpg \
  --prompt "Read the visible text and summarize the scene in two sentences." \
  --output-json outputs/gallery/ocr.json
```

For lower-memory GPUs, keep `--downsample-mode 16x` and reduce `--max-slice-nums` if needed. Use `--downsample-mode 4x` when you need finer visual detail and have enough memory.

## Multi-Image Comparison

Pass `--image-url` multiple times:

```bash
python examples/gallery/minicpmv46_inference.py \
  --image-url assets/airplane.jpeg \
  --image-url assets/worldmap_ck.jpg \
  --prompt "Compare these two images in two concise sentences." \
  --output-json outputs/gallery/multi_image.json
```

## Video Question Answering

Run a short video smoke test with a small frame budget:

```bash
python examples/gallery/minicpmv46_inference.py \
  --video-url https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4 \
  --prompt "Describe the main action in this video in one sentence." \
  --max-num-frames 8 \
  --max-slice-nums 1 \
  --output-json outputs/gallery/video.json
```

## Low-Memory Settings

The default examples use `--downsample-mode 16x`, which is the faster and lighter setting. If memory is tight, also reduce image slices and video frames:

```bash
python examples/gallery/minicpmv46_inference.py \
  --image-url assets/hk_OCR.jpg \
  --prompt "Read the visible text and summarize the scene." \
  --max-slice-nums 1 \
  --max-new-tokens 64 \
  --output-json outputs/gallery/ocr_low_memory.json
```

For more visual detail, switch to `--downsample-mode 4x` when your hardware has enough memory.

## Static Gallery

After writing one or more JSON outputs, build a local HTML gallery:

```bash
python examples/gallery/build_gallery.py \
  --result-json outputs/gallery/refraction.json \
  --result-json outputs/gallery/ocr.json \
  --result-json outputs/gallery/multi_image.json \
  --result-json outputs/gallery/video.json \
  --output outputs/gallery/index.html
```

Open `outputs/gallery/index.html` to switch between cases and inspect the input media, prompt, and model output.

## Local Gradio Demo

Launch a minimal local UI:

```bash
python examples/gallery/gradio_demo.py
```

The Gradio demo loads the model once, then accepts either an image or a video plus a prompt. Video inputs use the same frame-budget controls as the command-line script.
