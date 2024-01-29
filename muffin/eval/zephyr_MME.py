import os
import io
import glob
import torch
import base64
import pathlib
import argparse

from PIL import Image
from muffin.eval.zephyr_mm_chat import init_zephyr_mm, wrap_question_for_zephyr_mm


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def is_yes(text):
    if 'yes' in text.lower():
        return True
    else:
        return False


def eval_model(args):
    model, image_processor, image_token_len, tokenizer = init_zephyr_mm(
        args.model_name, tune_clip=args.tune_clip)

    if not pathlib.Path(args.out_dir).exists():
        pathlib.Path(args.out_dir).mkdir(parents=True)

    for task_path in list(glob.glob(f'{args.mme_dir}/*')):
        task_name = task_path.split('/')[-1]

        # if pathlib.Path(f'{args.out_dir}/{task_name}.txt').exists():
        #     continue
        ans_file_path = f'{args.out_dir}/{task_name}.txt'
        ans_file = open(ans_file_path, "w", encoding='utf-8')
        print(f'Write to {ans_file_path}')

        raw_ans_file = open(ans_file_path + '.jsonl', "w", encoding='utf-8')

        yes_cnt = 0
        no_cnt = 0
        for sample_path in list(glob.glob(f'{task_path}/*.txt')):

            lines = open(sample_path, encoding='utf-8').readlines()
            image_file = sample_path.replace('.txt', '.jpg')
            if not pathlib.Path(image_file).exists():
                image_file = sample_path.replace('.txt', '.png')

            for line in lines:
                qs = line.split('\t')[0]
                input_ids = torch.as_tensor(wrap_question_for_zephyr_mm(
                    qs, image_token_len=image_token_len, tokenizer=tokenizer)['input_ids'])
                input_token_len = input_ids.shape[0]

                if len(image_file) > 1000:
                    image = Image.open(io.BytesIO(
                        base64.b64decode(image_file))).convert('RGB')
                else:
                    image = Image.open(os.path.join(
                        args.image_folder, image_file)).convert('RGB')
                image_tensor = image_processor(image)

                with torch.inference_mode():
                    # print(f'Inference with prompt=>{tokenizer.decode(input_ids)}<=')
                    output = model.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        temperature=0.7,
                        max_new_tokens=12,
                        do_sample=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                        repetition_penalty=1.1)

                output_ids = output.sequences
                outputs = tokenizer.decode(
                    output_ids[0][input_token_len:], skip_special_tokens=True)
                # print(f'Output: {output_ids}@{outputs}@')

                outputs = outputs.strip()

                raw_ans_file.write(outputs + '\t' + image_file + '\n')
                raw_ans_file.flush()
                if is_yes(outputs):
                    outputs = 'Yes'
                    yes_cnt += 1
                else:
                    if 'no,' not in outputs.lower() and 'no.' not in outputs.lower() and outputs.lower() != 'no':
                        print(f'=> Output not yes/no: {outputs}', flush=True)
                    outputs = 'No'
                    no_cnt += 1

                out_line = f'{image_file.split("/")[-1]}\t{line.strip()}\t{outputs}\n'
                ans_file.write(out_line)
                ans_file.flush()
        print(f'{task_name}: Yes={yes_cnt}, No={no_cnt}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--mme_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="default")
    parser.add_argument('--tune_clip', type=str,
                        default='True', choices=['True', 'False'])
    args = parser.parse_args()

    args.tune_clip = True if args.tune_clip == 'True' else False
    eval_model(args)
