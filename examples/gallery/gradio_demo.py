import argparse
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from minicpmv46_inference import DEFAULT_PROMPT, load_model_and_processor, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a minimal local Gradio demo for MiniCPM-V 4.6.")
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def make_infer_fn(model, processor, base_args):
    def infer(image, video, prompt, max_new_tokens, downsample_mode, max_slice_nums, max_num_frames):
        image_url = None
        if image is not None:
            image_url = str(Path(image).resolve())
        video_url = None
        if video is not None:
            video_url = str(Path(video).resolve())

        args = SimpleNamespace(
            model_id=base_args.model_id,
            image_url=[image_url] if image_url and not video_url else None,
            video_url=video_url,
            prompt=prompt or DEFAULT_PROMPT,
            max_new_tokens=int(max_new_tokens),
            downsample_mode=downsample_mode,
            max_slice_nums=int(max_slice_nums),
            max_num_frames=int(max_num_frames),
            stack_frames=1,
        )
        answer, _ = run_inference(model, processor, args)
        return answer

    return infer


def main():
    args = parse_args()
    model, processor = load_model_and_processor(args)
    infer = make_infer_fn(model, processor, args)

    with gr.Blocks(title="MiniCPM-V 4.6 Local Demo") as demo:
        gr.Markdown("# MiniCPM-V 4.6 Local Demo")
        with gr.Row():
            image = gr.Image(type="filepath", label="Image")
            video = gr.Video(label="Video")
        prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt")
        with gr.Row():
            max_new_tokens = gr.Slider(16, 512, value=128, step=16, label="Max new tokens")
            downsample_mode = gr.Radio(["16x", "4x"], value="16x", label="Downsample mode")
            max_slice_nums = gr.Slider(1, 8, value=4, step=1, label="Max slice nums")
            max_num_frames = gr.Slider(1, 32, value=8, step=1, label="Max video frames")
        run = gr.Button("Run")
        output = gr.Textbox(label="Model output", lines=6)
        run.click(
            infer,
            inputs=[image, video, prompt, max_new_tokens, downsample_mode, max_slice_nums, max_num_frames],
            outputs=output,
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
