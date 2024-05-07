#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2024 AI, ZHIHU Inc. (zhihu.com)
#
# @author: chenqianyu <cqy1195@zhihu.com@zhihu.com>
# @date: 2024/5/03
#
import os
import glob
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import transformers
from trainer import CPMTrainer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed import zero

from dataset import data_collator, SupervisedDataset


from PIL import Image
from transformers import AutoModel, AutoTokenizer
from accelerate.utils import DistributedType

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, transform, data_collator=None, slice_config=None, 
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, transform, tokenizer, slice_config=slice_config)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, transform, tokenizer, slice_config=slice_config)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


local_rank = 0

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    
    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, torch_dtype=compute_dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    #Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, transform=model.transform,  data_collator=data_collator, slice_config=model.config.__dict__,
    )

    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()

