import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_IMAGE_URL = "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"
DEFAULT_VIDEO_URL = "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4"
DEFAULT_PROMPT = "Answer in one short sentence: what is shown in this input?"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal MiniCPM-V 4.6 inference example.")
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4.6", help="Hugging Face model id or local path.")
    parser.add_argument(
        "--image-url",
        action="append",
        help="Remote image URL or local image path. Pass multiple times for multi-image input.",
    )
    parser.add_argument("--video-url", help="Remote video URL or local video path. When set, image inputs are ignored.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt for the image or video.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--downsample-mode", default="16x", choices=["4x", "16x"])
    parser.add_argument("--max-slice-nums", type=int, default=4)
    parser.add_argument("--max-num-frames", type=int, default=16)
    parser.add_argument("--stack-frames", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None, help="Optional attention backend, e.g. flash_attention_2.")
    parser.add_argument("--local-files-only", action="store_true", help="Load model files from the local cache only.")
    parser.add_argument("--output-json", type=Path, help="Optional path to save the prompt and answer as JSON.")
    return parser.parse_args()


def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    return model, processor


def build_media_content(args):
    media_content = []
    if args.video_url:
        media_content.append({"type": "video", "url": args.video_url})
    else:
        for image_url in args.image_url or [DEFAULT_IMAGE_URL]:
            media_content.append({"type": "image", "url": image_url})
    return media_content


def run_inference(model, processor, args):
    media_content = build_media_content(args)
    messages = [{"role": "user", "content": media_content + [{"type": "text", "text": args.prompt}]}]

    processor_kwargs = {
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
    }
    if args.video_url:
        processor_kwargs.update(
            {
                "max_num_frames": args.max_num_frames,
                "stack_frames": args.stack_frames,
                "use_image_id": False,
            }
        )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            downsample_mode=args.downsample_mode,
            max_new_tokens=args.max_new_tokens,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer, media_content


def write_output_json(args, answer, media_content):
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "media": media_content,
                    "image_url": args.image_url or ([] if args.video_url else [DEFAULT_IMAGE_URL]),
                    "video_url": args.video_url,
                    "prompt": args.prompt,
                    "answer": answer,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def main():
    args = parse_args()
    model, processor = load_model_and_processor(args)
    answer, media_content = run_inference(model, processor, args)
    print(answer)
    write_output_json(args, answer, media_content)


if __name__ == "__main__":
    main()
