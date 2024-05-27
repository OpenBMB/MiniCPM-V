import sys
import datetime
import json
import os
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(script_dir, '..'))

from datasets.vqa_dataset import docVQADataset, docVQATESTDataset, textVQADataset


print(torch.__version__)

import numpy as np

from eval_utils.getargs import parse_args
from eval_utils.vqa_evaluate import *


def get_model(args):
    if args.model_name=='':
        raise Exception('Model name cannot be empty str!')
    from models.MiniCPM.minicpmv import MiniCPM_V
    model_path = args.model_path
    ckpt = args.ckpt
    model = MiniCPM_V(model_path=model_path, ckpt=ckpt, device=args.device)
    
    return model


def main(args):
    np.random.seed(0)
    max_sample_num = None

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    print(f'Init Rank-{torch.distributed.get_rank()}')
    if torch.distributed.is_initialized():
        args.device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model = get_model(args)
    
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if args.eval_textVQA or args.eval_all:
        dataset = textVQADataset(args.textVQA_image_dir, args.textVQA_ann_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time, \
                batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path)
        result['textVQA'] = acc

    if args.eval_docVQA or args.eval_all:
        dataset = docVQADataset(args.docVQA_image_dir, args.docVQA_ann_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'docVQA', time, batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path)
        result['docVQA'] = acc

    if args.eval_docVQATest or args.eval_all:
        target_dataset = "docVQATest"
        dataset = docVQATESTDataset(args.docVQATest_image_dir, args.docVQATest_ann_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path)
        result['docVQATest'] = acc
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    result_path = os.path.join(os.path.join(args.answer_path, args.model_name), 'result.json')
    
    output_flag = False
    for k, v in result.items():
        if v > 0.0:
            output_flag = True
            break
    
    if output_flag:
        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()

    main(args)