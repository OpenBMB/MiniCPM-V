import os
import math
import json
import copy
import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List
from PIL import Image


from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, raw_data, transform, tokenizer, slice_config):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        image = Image.open(self.raw_data[i]["image"]).convert("RGB")
        ret = preprocess(image, self.raw_data[i]["conversations"], self.tokenizer, self.transform, slice_config=self.slice_config)
        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["target"],
            attention_mask=ret["input_ids"].ne(self.tokenizer.pad_token_id),
            pixel_values=ret["pixel_values"],
            image_bound=ret["image_bound"],
        )
        
        return ret


def data_collator(examples, padding_value=0):
    input_ids = pad_sequence([example["input_ids"] for example in examples], batch_first=True, padding_value=padding_value)
    targets = pad_sequence([example["labels"] for example in examples], batch_first=True, padding_value=padding_value)
    attention_mask = pad_sequence([example["attention_mask"] for example in examples], batch_first=True, padding_value=padding_value)
    pixel_values = [example["pixel_values"] for example in examples]
    image_bound = [example["image_bound"] for example in examples]
    return {"input_ids": input_ids, "labels":targets, "attention_mask": attention_mask, "image_bound": image_bound, "pixel_values": pixel_values}


def conversation_to_ids(conversation, tokenizer):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    raw_msg = ''
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg['role']
        message = msg['content']
        assert role in ['user', 'assistant']
        if role == 'user':
            prefix = '<用户>'
        else:
            prefix = '<AI>'
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:] # remove bos
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == 'assistant':
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += (prefix + message)
    
    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))

    # build target
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            target[i - 1] = tokenizer.eos_id

    # build image bound
    image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        print('image start token != image end tokens')
    if len(image_start_tokens)>0:
        image_bound = torch.hstack([image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)])
    else:
        image_bound = []

    return {
        'input_ids': ids,
        'target': target,
        'image_bound': image_bound,
        'raw_msg': raw_msg,
    }


def preprocess(image, conversation, tokenizer, transform, query_nums=64, slice_config=None):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    assert len(conversation) > 1, "conversation length must large than 2"
    assert conversation[0]['role'] == 'user', "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert 'patch_size' in slice_config
        assert 'max_slice_nums' in slice_config
        assert 'scale_resolution' in slice_config
    default_image_placeholder = tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    if slice_config:
        images = []
        source_image, patches, best_grid = slice_image(
            image, slice_config['max_slice_nums'], slice_config['scale_resolution'], slice_config['patch_size']
        )
        images.append(source_image)
        image_placeholder = default_image_placeholder
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    images.append(patches[i][j])

            image_placeholder += get_grid_placeholder(
                tokenizer, best_grid, query_nums
            )
        images = [transform(i) for i in images]
    else:
        images = [transform(image)]
        image_placeholder = default_image_placeholder
    if '<image>' in conversation[0]['content']:
        conversation[0]['content'] = conversation[0]['content'].replace('<image>', image_placeholder)
    else:
        conversation[0]['content'] = image_placeholder + '\n' + conversation[0]['content']

    input_dict = conversation_to_ids(conversation, tokenizer)
    input_dict['pixel_values'] = images
    return input_dict



def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder

