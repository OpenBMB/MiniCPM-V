import os
import gc
import copy
import time

import torch
import warnings
import transformers

import numpy as np

from typing import Dict, Optional, Sequence
from omnilmm import conversation as conversation_lib

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def SFT_collator_fn(instances, pad_token_id):
    input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        # attention_mask=input_ids.ne(pad_token_id),
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
    return batch


def encode_multimodal_preference_sample(source, tokenizer, multimodal_cfg, preprocess_func=None):
    if isinstance(source['chosen'], list):
        win_conv = source['chosen']
        rej_conv = source['rejected']
    elif isinstance(source['chosen'], dict):
        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

    if 'image' in source:
        image = source['image']
        image = multimodal_cfg['image_processor'](image)
        win_conv = expand_image_token(win_conv, multimodal_cfg)
        rej_conv = expand_image_token(rej_conv, multimodal_cfg)

    if preprocess_func is None:
        rej_data_dict = preprocess([rej_conv], tokenizer)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                             labels=rej_data_dict["labels"][0])

        win_data_dict = preprocess([win_conv], tokenizer)
        win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                             labels=win_data_dict["labels"][0])
    else:
        rej_data_dict = preprocess_func([rej_conv], tokenizer)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                             labels=rej_data_dict["labels"][0])
        win_data_dict = preprocess_func([win_conv], tokenizer)
        win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                             labels=win_data_dict["labels"][0])

    # image exist in the data
    if 'image' in source:
        rej_data_dict['image'] = win_data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        rej_data_dict['image'] = win_data_dict['image'] = torch.zeros(
            3, crop_size['height'], crop_size['width'])

    if 'ref_win_logp' in source:
        rej_data_dict['ref_rej_logp'] = source['ref_rej_logp']
        win_data_dict['ref_win_logp'] = source['ref_win_logp']
        rej_data_dict['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
        win_data_dict['ref_win_avg_logp'] = source['ref_win_avg_logp']
        rej_data_dict['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        win_data_dict['ref_win_per_token_logp'] = source['ref_win_per_token_logp']
    return rej_data_dict, win_data_dict


def expand_image_token(source, multimodal_cfg) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    image_token_len = multimodal_cfg['image_token_len']
    if not is_multimodal:
        return source

    for sentence in source:
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        if multimodal_cfg['use_im_start_end']:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        sentence["value"] = sentence["value"].replace(
            DEFAULT_IMAGE_TOKEN, replace_token)

    return source


def omni_encode_multimodal_sample(source, tokenizer, multimodal_cfg):
    conversation = copy.deepcopy(source["conversations"])
    if 'image' in source:
        image = source['image']
        image = multimodal_cfg['image_processor'](image)
        conversation = expand_image_token(conversation, multimodal_cfg)

    data_dict = omni_preprocess([conversation], tokenizer)
    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])

    ##
    # source_id = source['metainfo']['origin_idx']
    # input_ids = data_dict['input_ids']
    # print(f'@==>{source_id}<==@', flush=True)
    # import re
    # print(f'@==>{source_id} inputs', re.sub('<im_patch>', '', tokenizer.decode(input_ids)), flush=True)
    # print(f'@==>{source_id} Inputs', tokenizer.decode(input_ids), flush=True)
    # print(f'@==>{source_id} Input Tokens', tokenizer.convert_ids_to_tokens(input_ids), flush=True)

    # labels = data_dict['labels']
    # labels = [(1 if x.item() == -100 else x) for x in labels]
    # print('labels', tokenizer.decode(labels), flush=True)
    # print(f'@==>{source_id} Labels', tokenizer.decode(labels), flush=True)
    # print(f'@==>{source_id} Label Tokens', tokenizer.convert_ids_to_tokens(labels), flush=True)
    ##

    # image exist in the data
    if 'image' in source:
        data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        raise NotImplementedError
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        data_dict['image'] = torch.zeros(
            3, crop_size['height'], crop_size['width'])
    return data_dict


def encode_multimodal_sample(source, tokenizer, multimodal_cfg):
    conversation = copy.deepcopy(source["conversations"])
    if 'image' in source:
        image = source['image']
        image = multimodal_cfg['image_processor'](image)
        conversation = expand_image_token(conversation, multimodal_cfg)

    data_dict = preprocess([conversation], tokenizer)
    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])

    # image exist in the data
    if 'image' in source:
        data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        data_dict['image'] = torch.zeros(
            3, crop_size['height'], crop_size['width'])
    return data_dict


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def omni_preprocess(sources,
                      tokenizer: transformers.PreTrainedTokenizer,
                      generation=False):
    system_content = 'You are an artificial intelligence assistant, which gives helpful, detailed, and polite answers to the human\'s questions.'
    ignore_index = -100

    response_template = '\n<|assistant|>\n'
    instruction_template = '\n<|user|>\n'
    response_token_ids = tokenizer.encode(
        response_template, add_special_tokens=False)
    instruction_token_ids = tokenizer.encode(
        instruction_template, add_special_tokens=False)

    batch_input_ids = []
    batch_labels = []
    for i in range(len(sources)):
        new_source = []
        prev_role = 'unexpect'
        for conv_turn in sources[i]:
            role = conv_turn['from'] if 'from' in conv_turn else conv_turn['role']
            content = conv_turn['value'] if 'value' in conv_turn else conv_turn['content']

            role = 'user' if role == 'human' else role
            role = 'assistant' if role == 'gpt' else role

            assert role in ['user', 'assistant']
            assert role != prev_role, f'role={role}, prev_role={prev_role}'
            prev_role = role

            new_turn = {
                'role': role,
                'content': content
            }
            new_source.append(new_turn)
        if new_source[0]['role'] != 'system':
            new_source.insert(0, {'role': 'system', 'content': system_content})

        # TODO: this automatically add '\n' to the end
        res_text = tokenizer.apply_chat_template(
            new_source, tokenize=False, add_generation_prompt=generation)
        if not generation:
            res_text = res_text.strip()

        conversations_tokenized = _tokenize_fn([res_text], tokenizer)
        res_input_ids = conversations_tokenized["input_ids"][0]

        # since labels and input_ids are reference towards the same object
        res_labels = copy.deepcopy(conversations_tokenized["labels"][0])

        response_token_ids_idxs = []
        human_token_ids_idxs = []

        for assistant_idx in np.where(res_labels == response_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (response_token_ids == res_labels[assistant_idx: assistant_idx + len(
                        response_token_ids)].tolist()
                    ):
                response_token_ids_idxs.append(
                    assistant_idx + len(response_token_ids))

        if len(response_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{response_template}` in the "
                f'following instance: @===>{tokenizer.decode(res_input_ids)}<===@ '
                f'Raw text is @===>{res_text}<===@'
                f'Raw source is @===>{new_source}<===@'
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            res_labels[:] = ignore_index

        human_token_ids = instruction_token_ids
        for human_idx in np.where(res_labels == human_token_ids[0])[0]:
            # find the indexes of the start of a human answer.
            if human_token_ids == res_labels[human_idx: human_idx + len(human_token_ids)].tolist():
                human_token_ids_idxs.append(human_idx)

        if len(human_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find instruction key `{instruction_template}` in the "
                f'following instance: @===>{tokenizer.decode(res_input_ids)}<===@ '
                f'Raw text is @===>{res_text}<===@'
                f'Raw source is @===>{new_source}<===@'
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            res_labels[:] = ignore_index

        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            # Make pytorch loss function ignore all non response tokens
            if idx != 0:
                res_labels[start:end] = ignore_index
            else:
                res_labels[:end] = ignore_index

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            res_labels[human_token_ids_idxs[-1]:] = ignore_index

        batch_input_ids.append(res_input_ids)
        batch_labels.append(res_labels)

    return dict(input_ids=batch_input_ids, labels=batch_labels)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
