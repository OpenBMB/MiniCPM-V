import io
import re
import glob
import math
import json
import base64
import random
import copy

from PIL import Image
from typing import List


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(keys, value):
            if not callable(value):
                raise Exception(
                    f"Register object must be callable! But receice:{value} is not callable!")

            if not isinstance(keys, list):
                keys = [keys]

            for key in keys:
                if key in self._dict:
                    print(
                        f"error: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
                    exit()

                self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x: add_register_item(target, x)

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


register_data_processor = Register()
register_data_path = Register()


def vqa_instruction_templates(question, idx=None):
    instructions = [
        "{Question} A short answer to the question is",
        "Given the image, answer the following question with no more than three words. {Question}",
        "Based on the image, respond to this question with a short answer: {Question} Answer:",
        "Use the provided image to answer the question: {Question} Provide your answer as short as possible:",
    ]
    if idx is None:
        new_question = random.choice(
            instructions).replace("{Question}", question)
    else:
        new_question = instructions[idx].replace("{Question}", question)

    return new_question


def caption_instruction_templates():
    instructions = [
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the subsequent image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo's key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."
    ]

    new_question = random.choice(instructions)

    return new_question


def ocr_instruction_templates():
    instructions = [
        "Identify the text in the image with position."
        "Pinpoint and indicate the text and its location within the image."
        "Find the text in the image and identify its positional."
        "Detect the text within the image and specify its position."
        "Locate the text in the image and detail its position."
    ]

    new_question = random.choice(instructions)

    return new_question


def textvqa_instruction_templates(question):
    instructions = [
        "Answer the question shortly by reading the texts. {Question}"
        "After reading the text in the image, {Question} A short answer to the question is",
        "Given the text in the image, answer the following question with no more than three words. {Question}"
    ]

    new_question = random.choice(instructions).replace("{Question}", question)

    return new_question


def load_multimodal_conversation(text_b64, img_b64_buffer):
    map_role = {
        'human': 'human',
        'gpt': 'gpt'
    }

    text = base64.b64decode(text_b64).decode('utf-8')
    list_conv = json.loads(text)

    out: List[dict] = []
    for idx, sentence in enumerate(list_conv):
        value = sentence['value']

        if idx == 0 and '<image>' not in value:
            value = f"<image>\n{value}"
        if idx != 0 and '<image>' in value:
            value = value.replace('<image>', '')

        out.append({
            'from': map_role[sentence['from']],
            'value': value
        })

    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image, out


def load_caterpillar_multimodal_conversation(text_b64, img_b64_buffer):
    map_role = {
        'human': 'human',
        'gpt': 'gpt'
    }

    text = base64.b64decode(text_b64).decode('utf-8')
    list_conv = json.loads(text)

    out: List[dict] = []
    for idx, sentence in enumerate(list_conv):
        value = sentence['value']
        value = re.sub(r'<image>.+?</image>', '<image>', value)

        out.append({
            'from': map_role[sentence['from']],
            'value': value.strip()
        })

    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image, out


def load_pretrain_conversation(text_b64, img_b64_buffer):
    map_role = {
        'human': 'human',
        'gpt': 'gpt'
    }

    text = base64.b64decode(text_b64).decode('utf-8')
    list_conv = json.loads(text)

    out: List[dict] = []
    for idx, sentence in enumerate(list_conv):
        print(sentence)
        value = sentence['value']
        value = re.sub(r'<image>.+?</image>', '<image>', value)

        out.append({
            'from': map_role[sentence['from']],
            'value': value.strip()
        })

    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image, out


def b64_to_PIL_image(img_b64_buffer):
    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image


def wrap_qa_to_single_turn_multimodal_conv(answer, question):
    if '<image>' not in question:
        question = f"<image>\n{question}"

    out = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer}
    ]
    return question, out


def wrap_generation_single_turn_conv(out, template_func):
    conv = [
        {
            "from": "human",
            "value": f"<image>\n{template_func()}"

        },
        {
            "from": "gpt",
            "value": out
        }
    ]
    return conv


def wrap_ocr_generation_single_turn_conv(out):
    return wrap_generation_single_turn_conv(out, ocr_instruction_templates)


def wrap_caption_generation_single_turn_conv(out):
    return wrap_generation_single_turn_conv(out, caption_instruction_templates)


def gather_data_files_by_glob(root: str, pattern='*.tsv'):
    filenames = []

    for fullpath in glob.glob(f'{root}/{pattern}'):
        filename = fullpath.split('/')[-1]
        filenames.append(filename)
    return root, filenames


@register_data_path('caterpillar-stage1')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage1'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage2_grounding')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage2_grounding'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage2_mix')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage2_mix'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage2_ocr')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage2_ocr'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage2_ocr_zh')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage2_ocr_zh'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage2_ocr_en')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage2_ocr_en'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage3_lvis')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage3/lvis-instruct4v'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage3_svit')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage3/svit'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage3_svit_nobbox')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage3/svit_nobbox'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('caterpillar-stage3_sharegpt4v')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage3/sharegpt4v'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('pretrain_eval_1213')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/pretrain_eval_tsv/1213_eval_tsv_has_ds'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('eval_mmbench')
def caterpillar_stage1_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/pretrain_eval_tsv/eval_mmbench_has_ds'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_processor(['caterpillar-stage1', 'caterpillar-stage2_grounding', 'caterpillar-stage2_ocr_en',
                          'caterpillar-stage2_ocr', 'caterpillar-stage3_lvis',
                          'caterpillar-stage3_svit', 'caterpillar-stage3_svit_nobbox',
                          'caterpillar-stage3_sharegpt4v',
                          'pretrain_eval_1213', 'eval_mmbench'])
def caterpillar_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft' or intent == 'eval':
        image, out = load_caterpillar_multimodal_conversation(
            text_b64, img_b64_buffer)

        if intent != 'eval':
            refine_out = []
            for i in range(0, len(out), 2):
                question = out[i]['value']
                if 'Answer the question directly with a short sentence or phrase.' in question:
                    answer = out[i+1]['value']
                    ans_len = len(answer.split())
                    if ans_len <= 3:
                        question = question.replace('Answer the question directly with a short sentence or phrase.',
                                                    'Please answer the question using a single word or phrase.')
                refine_out.append({
                    'from': out[i]['from'],
                    'value': question
                })

                refine_out.append(out[i+1])

            out = refine_out

        metainfo = {
            "origin_dataset": origin_dataset,  # parquet folder name
            "origin_split": origin_split,  # parquet file name
            "origin_idx": origin_split_inner_idx,  # index in origin parquet file
            "image_id": img_path,  # image_id
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_processor('caterpillar-stage2_ocr_zh')
def caterpillar_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_caterpillar_multimodal_conversation(
            text_b64, img_b64_buffer)

        question = out[0]['value']
        if not question.replace('<image>', '').strip().endswith('。'):
            if question.endswith('>'):
                question = question.split('\n')[0] + '。\n' + '<image>'
            else:
                question = question + '。'

            out[0]['value'] = question

        metainfo = {
            "origin_dataset": origin_dataset,  # parquet folder name
            "origin_split": origin_split,  # parquet file name
            "origin_idx": origin_split_inner_idx,  # index in origin parquet file
            "image_id": img_path,  # image_id
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_processor(['caterpillar-stage2_mix'])
def caterpillar_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_caterpillar_multimodal_conversation(
            text_b64, img_b64_buffer)

        refine_out = []
        for i in range(0, len(out)):
            text = out[i]['value']
            if text.endswith("BECAUSE:"):
                new_len = len(text) - len("BECAUSE:")
                text = text[:new_len].strip()

            refine_out.append({
                'from': out[i]['from'],
                'value': text
            })

        out = refine_out

        metainfo = {
            "origin_dataset": origin_dataset,  # parquet folder name
            "origin_split": origin_split,  # parquet file name
            "origin_idx": origin_split_inner_idx,  # index in origin parquet file
            "image_id": img_path,  # image_id
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented

@register_data_path('caterpillar_mix')
def caterpillar_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/caterpillar_SFT/tsv_2/caterpillar/stage123'
    root_dir, filenames = gather_data_files_by_glob(data_dir, '*.tsv')
    filenames = sorted(filenames)
    return root_dir, filenames

@register_data_processor('caterpillar_mix')
def caterpillar_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        if 'vqa_chat_20230628' in origin_dataset or 'LLaVA_Instruct_150K' in origin_dataset:
            image, out = load_multimodal_conversation(text_b64, img_b64_buffer)
        else:
            image, out = load_caterpillar_multimodal_conversation(text_b64, img_b64_buffer)

        refine_out = []
        for i in range(0, len(out), 2):
            question = out[i]['value']
            if 'Answer the question directly with a short sentence or phrase.' in question:
                answer = out[i+1]['value']
                ans_len = len(answer.split())
                if ans_len <= 3:
                    question = question.replace('Answer the question directly with a short sentence or phrase.',
                                                'Please answer the question using a single word or phrase.')
            refine_out.append({
                'from': out[i]['from'],
                'value': question
            })

            ### deal with no rationale data
            text = out[i+1]['value']
            if text.endswith("BECAUSE:"):
                new_len = len(text) - len("BECAUSE:")
                text = text[:new_len].strip()

            refine_out.append({
                'from': out[i+1]['from'],
                'value': text
            })

        out = refine_out

        metainfo = {
            "origin_dataset": origin_dataset,  # parquet folder name
            "origin_split": origin_split,  # parquet file name
            "origin_idx": origin_split_inner_idx,  # index in origin parquet file
            "image_id": img_path,  # image_id
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('unimm-chat')
def unimmchat_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/sft_data/coco_based/vqa_chat_20230628/'
    return gather_data_files_by_glob(data_dir, '*.tsv')

@register_data_path('ultrachat_200k')
def text_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/text_only_data/ultrachat_200k/train_sft/'
    return gather_data_files_by_glob(data_dir, '*.tsv')

@register_data_path('ultrachat_200k_no_long')
def text_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/text_only_data/ultrachat_200k/train_sft_filter_long/'
    return gather_data_files_by_glob(data_dir, '*.tsv')

@register_data_path('alpaca')
def text_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/text_only_data/alpaca'
    return gather_data_files_by_glob(data_dir, '*.tsv')

@register_data_path('sharegpt')
def text_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/text_only_data/sharegpt/clean'
    return gather_data_files_by_glob(data_dir, '*.tsv')

@register_data_processor(['unimm-chat', 'pretrain_eval_train', 'pretrain_eval_eval',
                          'ultrachat_200k', 'ultrachat_200k_no_long', 'alpaca', 'sharegpt'])
def unimmchat_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft' or intent == 'eval':
        if img_b64_buffer == '<no_image>':
            image = '<no_image>'
            out = base64.b64decode(text_b64).decode('utf-8')
            out = json.loads(out)
        else:
            image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # unimm-chat folder name
            "origin_split": origin_split,  # unimm-chat parquet file name
            "origin_idx": origin_split_inner_idx,  # index in unimm-chat parquet file
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('gpt4v_detailed_1102')
def gpt4v_detailed_1102_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/SFT_gpt4v_detailed_20231102'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('gpt4v_detailed_1105')
def gpt4v_detailed_1105_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/SFT_gpt4v_detailed_20231105'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_processor('gpt4v_detailed_1105')
def gpt4v_detailed_1105_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('dpo_sftwin_1103-1106')
def dpo_sftwin_1103_1106_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_win_sft_1103-1106'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_processor('dpo_sftwin_1103-1106')
def dpo_sftwin_1103_1106_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('dpo_sftwin_1005-1026')
def dpo_sftwin_1005_1026_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_win_sft_1005-1026'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_processor('dpo_sftwin_1005-1026')
def dpo_sftwin_1005_1026_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('dpo_sftwin_checked_1005-1026')
def dpo_sftwin_1005_1026_data_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/DPO_win_sft_human_inspected"
    return gather_data_files_by_glob(data_dir, '*checked_description*.tsv')


@register_data_processor('dpo_sftwin_checked_1005-1026')
def dpo_sftwin_checked_1005_1026_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('dpo_sftwin_checked_1103-1106')
def dpo_sftwin_checked_1103_1106_data_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/DPO_win_sft_human_inspected"
    return gather_data_files_by_glob(data_dir, '*checked_qa*.tsv')


@register_data_processor('dpo_sftwin_checked_1103-1106')
def dpo_sftwin_checked_1103_1106_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('omnilmm_cvpr_rewrite_bytrans_sft')
def dpo_sft_ncrp_vqa_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_cvpr"
    return gather_data_files_by_glob(data_dir, pattern='omnilmm_rewrite-by_translate_cvpr_dpo_with_per_token_vqa_logp_train-1401.tsv')


@register_data_processor('omnilmm_cvpr_rewrite_bytrans_sft')
def dpo_sft_ncrp_vqa_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('omnilmm_rewrite-by_trans_cvpr_1020_1027-1124_sft')
def dpo_sft_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_cvpr_1020_1027-1124_good/'
    return gather_data_files_by_glob(data_dir, pattern='omnilmm_rewrite-by_trans_cvpr_zj1020_zj1027-1124_dpo_with_per_token_vqa_logp_train-2102.tsv')


@register_data_processor('omnilmm_rewrite-by_trans_cvpr_1020_1027-1124_sft')
def dpo_sft_ncrp_vqa_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_path('llava_rewrite-by_trans_1122-1123_1128_sft')
def dpo_sft_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_llava_1122-1123_1128/'
    return gather_data_files_by_glob(data_dir, pattern='llava_rewrite-by_trans_zj1122-1123_1128_good_dpo_with_per_token_vqa_logp_train-1065.tsv')


@register_data_processor('llava_rewrite-by_trans_1122-1123_1128_sft')
def dpo_sft_ncrp_vqa_processor(*args, **kwargs):
    return gpt4v_detailed_1102_processor(*args, **kwargs)


@register_data_processor('gpt4v_detailed_1102')
def gpt4v_detailed_1102_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                  intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        question = list_conv[0]
        if '<image>' not in question:
            question = f"<image>\n{question}"

        out_ans = list_conv[1]

        question = {"from": "human", "value": question}
        out_ans = {"from": "gpt", "value": out_ans}

        out = [question, out_ans]

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_idx": origin_split_inner_idx,
            "origin_split": origin_split,  # metainfos
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('llava')
def llava_instruct_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/LLaVA/'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_processor('llava')
def llava_instruct_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                             intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # llava folder name
            "origin_split": origin_split,  # llava parquet file name
            "origin_idx": origin_split_inner_idx,  # index in llava parquet file
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('m3it')
def m3it_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/M3IT'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('m3it')
def m3it_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                   intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # m3it
            "origin_split": origin_split,  # m3it parquet file name
            "origin_idx": origin_split_inner_idx,  # index in m3it parquet file
            "image_id": img_path,  # image_path in m3it parquet
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('lrv_clean')
def lrv_clean_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/LRV'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('lrv_clean')
def lrv_clean_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # LRV
            "origin_split": origin_split,  # LRV parquet file name, including pos/neg tag
            "origin_idx": origin_split_inner_idx,  # index in LRV parquet file
            "image_id": img_path,  # VisualGenome image id
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('synthdog_yesno_shuffle')
def yesno_synthdog_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/sft_data/yesno_data_shuffle/yesno_ocr'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('synthdog_yesno_shuffle')
def yesno_synthdog_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                             intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # yesno folder name
            "origin_split": origin_split,  # yesno parquet file name
            "origin_idx": origin_split_inner_idx,  # index in yesno parquet file
            "image_id": img_path,  # synthdog parquet filename and item index
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('vqa_yesno_shuffle')
def yesno_vqa_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/sft_data/yesno_data_shuffle/yesno_vqa'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('vqa_yesno_shuffle')
def yesno_vqa_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # yesno folder name
            "origin_split": origin_split,  # yesno parquet file name
            "origin_idx": origin_split_inner_idx,  # index in yesno parquet file
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('synthdog_yesno_suffix_shuffle')
def yesno_synthdog_suffix_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/sft_data/yesno_data_suffix_shuffle/yesno_ocr_suffix'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('synthdog_yesno_suffix_shuffle')
def yesno_synthdog_suffix_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                    intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # yesno folder name
            "origin_split": origin_split,  # yesno parquet file name
            "origin_idx": origin_split_inner_idx,  # index in yesno parquet file
            "image_id": img_path,  # synthdog parquet filename and item index
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('vqa_yesno_suffix_shuffle')
def yesno_vqa_suffix_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/sft_data/yesno_data_suffix_shuffle/yesno_vqa_suffix'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('vqa_yesno_suffix_shuffle')
def yesno_vqa_suffix_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                               intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # yesno folder name
            "origin_split": origin_split,  # yesno parquet file name
            "origin_idx": origin_split_inner_idx,  # index in yesno parquet file
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


# 1401: cvpr
# 2102: cvpr + new omnilmm data
# 1065: llava outputs
# 2566: multi_model outputs [1218, 1229]
@register_data_processor(['dpo_omni_1401-en', 'dpo_omni_1401-trs', 'dpo_omni_2102-en', 'dpo_omni_2102-trs', 'dpo_omni_1065-trs', 'dpo_omni_2566-trs', 'RM_Bench_clean_diff1', 'RM_Bench_clean_diff2', 'RM_Bench_clean_diff3'])
def dpo_data_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)

@register_data_processor(['dpo_omni_2102-trs_SFT', 'dpo_omni_1065-trs_SFT', 'dpo_omni_2566-trs_SFT', 'RM_Bench_clean_diff1_SFT', 'RM_Bench_clean_diff2_SFT', 'RM_Bench_clean_diff3_SFT'])
def dpo_data_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_omni_1401-en')
def dpo_data_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/refined_test/"
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp-1401.tsv')

@register_data_path('dpo_omni_1401-trs')
def dpo_data_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/refined_test/"
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_by_trans-1401.tsv')

@register_data_path('dpo_omni_2102-en')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_cvpr_1020_1027-1124_good/'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp-2102.tsv')

@register_data_path('dpo_omni_2102-trs')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_cvpr_1020_1027-1124_good/'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_by_trans-2102.tsv')

@register_data_path('dpo_omni_2102-trs_SFT')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_cvpr_1020_1027-1124_good/'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_SFT_by_trans-2102.tsv')

@register_data_path('dpo_omni_1065-trs')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_llava_1122-1123_1128/'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_by_trans-1065.tsv')

@register_data_path('dpo_omni_1065-trs_SFT')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_llava_1122-1123_1128/'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_SFT_by_trans-1065.tsv')

@register_data_path('dpo_omni_2566-trs')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/DPO_diverse_20231218-1229_all/"
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_by_trans-2566.tsv')

@register_data_path('dpo_omni_2566-trs_SFT')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = "/data/public/multimodal/multimodal_data/dpo/DPO_diverse_20231218-1229_all/"
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_SFT_by_trans-2566.tsv')

@register_data_path('RM_Bench_clean_diff1')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp-893.tsv')

@register_data_path('RM_Bench_clean_diff2')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp-262.tsv')

@register_data_path('RM_Bench_clean_diff3')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp-90.tsv')

@register_data_path('RM_Bench_clean_diff1_SFT')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_SFT-893.tsv')

@register_data_path('RM_Bench_clean_diff2_SFT')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_SFT-262.tsv')

@register_data_path('RM_Bench_clean_diff3_SFT')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    return gather_data_files_by_glob(data_dir, pattern='omni_stage3_4k_logp_SFT-90.tsv')



@register_data_path('dpo_1005')
def dpo_preference_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_logp_train*.tsv')


@register_data_path('dpo_preference_1012')
def dpo_preference_data_1012_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231012'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_logp_train*.tsv')


@register_data_path('dpo_preference_llava_7b_v1_preference_hallonly')
def dpo_preference_data_llava_rlhf_hallonly_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_hallonly'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_logp_train*.tsv')


@register_data_path('dpo_preference_llava_7b_v1_preference_hallonly_nocontext_no_crop_vqa')
def dpo_preference_data_llava_rlhf_hallonly_nocontext_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_hallonly_nocontext'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_llava_7b_v1_preference_nothall')
def dpo_preference_data_llava_rlhf_nothall_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_nothall'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_logp_train*.tsv')


@register_data_path('dpo_preference_llava_7b_v1_preference_nothall_nocontext_no_crop_vqa')
def dpo_preference_data_llava_rlhf_nothall_nocontext_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_nothall_nocontext'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_ncrp_vqa_1005')
def dpo_preference_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_llava')
def dpo_preference_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_llava_logp_train*.tsv')


@register_data_path('dpo_preference_1012_no_crop_vqa')
def dpo_preference_1012_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231012'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1012_llava')
def dpo_preference_1012_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231012'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_llava_logp_train*.tsv')


@register_data_path('dpo_preference_1013_no_crop_vqa')
def dpo_preference_1013_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231013'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1016_ncrp_vqa')
def dpo_preference_1016_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231016'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1017_no_crop_vqa')
def dpo_preference_1017_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231017'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1018_no_crop_vqa')
def dpo_preference_1018_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231018'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1019_ncrp_vqa')
def dpo_preference_1019_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231019'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1020_no_crop_vqa')
def dpo_preference_1020_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231020'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1023_ncrp_vqa')
def dpo_1023_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231023'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1024_ncrp_vqa')
def dpo_1024_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231024'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1025_ncrp_vqa')
def dpo_preference_1025_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231025-new'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1026_ncrp_vqa')
def dpo_preference_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231026-new'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1103-1106_ncrp_vqa')
def dpo_preference_1103_1106_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231103-1to1106-3'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1012-1127_ncrp_vqa_bad')
def dpo_preference_1012_1127_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1012-1127_bad_refine'
    return gather_data_files_by_glob(data_dir, pattern='*.tsv')


@register_data_processor('dpo_1012-1127_ncrp_vqa_bad')
def dpo_cvpr_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1027-1124_ncrp_vqa_good')
def dpo_preference_1027_1124_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1027-1124_good_refine'
    return gather_data_files_by_glob(data_dir, pattern='*.tsv')


@register_data_processor('dpo_1027-1124_ncrp_vqa_good')
def dpo_cvpr_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1020_ncrp_vqa_good')
def dpo_preference_1020_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1020_good_refine'
    return gather_data_files_by_glob(data_dir, pattern='*.tsv')


@register_data_processor('dpo_1020_ncrp_vqa_good')
def dpo_cvpr_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_ncrp_vqa_after_SFT_with_VQA')
def dpo_preference_1103_1106_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_after_SFT_with_vqa_logp_train-1401.tsv')


@register_data_processor('dpo_cvpr_ncrp_vqa_after_SFT_with_VQA')
def dpo_1103_1106_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1107_ncrp_vqa')
def dpo_preference_1103_1106_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231107/'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_processor('dpo_1107_ncrp_vqa')
def dpo_1005_1026_g_4v1103_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_docrp_vqa')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_docrop_vqa_logp_train-1401.tsv')


@register_data_processor('dpo_cvpr_docrp_vqa')
def dpo_cvpr_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_ncrp_vqa')
def dpo_cvpr_ncrp_vqa_path():
    # data_dir = '/home/zhanghaoye/apps/supporting_repo/Muffin/data/RLHF-V-Hall_v0/'
    # return gather_data_files_by_glob(data_dir, pattern='RLHF-V-Hall_v0_dpo_with_rlhf-v-sft_logp_train-1401.tsv')
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-1401.tsv')


@register_data_processor('dpo_cvpr_ncrp_vqa')
def dpo_cvpr_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_800_ncrp_vqa')
def dpo_cvpr_800_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-800.tsv')


@register_data_processor('dpo_cvpr_800_ncrp_vqa')
def dpo_cvpr_800_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_400_ncrp_vqa')
def dpo_cvpr_400_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-400.tsv')


@register_data_processor('dpo_cvpr_400_ncrp_vqa')
def dpo_cvpr_400_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_200_ncrp_vqa')
def dpo_cvpr_200_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-200.tsv')


@register_data_processor('dpo_cvpr_200_ncrp_vqa')
def dpo_cvpr_200_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_100_ncrp_vqa')
def dpo_cvpr_100_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-100.tsv')


@register_data_processor('dpo_cvpr_100_ncrp_vqa')
def dpo_cvpr_100_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llava')
def dpo_cvpr_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-1401.tsv')


@register_data_path('dpo_cvpr_after_SFT_llava')
def dpo_cvpr_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_after_SFT_logp_train-1401.tsv')


@register_data_processor('dpo_cvpr_after_SFT_llava')
def dpo_cvpr_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_cvpr_llava')
def dpo_cvpr_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_800_llava')
def dpo_cvpr_800_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-800.tsv')


@register_data_processor('dpo_cvpr_800_llava')
def dpo_cvpr_800_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_400_llava')
def dpo_cvpr_400_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-400.tsv')


@register_data_processor('dpo_cvpr_400_llava')
def dpo_cvpr_400_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_200_llava')
def dpo_cvpr_200_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-200.tsv')


@register_data_processor('dpo_cvpr_200_llava')
def dpo_cvpr_200_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_100_llava')
def dpo_cvpr_100_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main/'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-100.tsv')


@register_data_processor('dpo_cvpr_100_llava')
def dpo_cvpr_100_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_llava')
def dpo_cvpr_llavarlhf_onlyhall_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-2122.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_llava')
def dpo_cvpr_llavarlhf_onlyhall_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_800_llava')
def dpo_cvpr_llavarlhf_onlyhall_800_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-800.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_800_llava')
def dpo_cvpr_llavarlhf_onlyhall_800_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_200_llava')
def dpo_cvpr_llavarlhf_onlyhall_200_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_llava_logp_train-200.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_200_llava')
def dpo_cvpr_llavarlhf_onlyhall_200_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-2122.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_800_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_800_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-800.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_800_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_800_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_400_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_400_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-400.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_400_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_400_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_200_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_200_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-200.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_200_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_200_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_cvpr_llavarlhf_onlyhall_100_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_100_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    return gather_data_files_by_glob(data_dir, pattern='dpo_with_per_token_vqa_logp_train-100.tsv')


@register_data_processor('dpo_cvpr_llavarlhf_onlyhall_100_ncrp_vqa')
def dpo_cvpr_llavarlhf_onlyhall_100_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_1005-1026_ncrp_vqa')
def dpo_preference_1103_1106_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1005-1026_good'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1027-1101-2_ncrp_vqa')
def dpo_preference_1027_1101_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231027to1101-2'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_checked_1005-1026_ncrp_vqa')
def dpo_preference_checked_1005_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_human_inspected_1005-1026'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_checked_1103-1106_ncrp_vqa')
def dpo_preference_checked_1103_1106_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_human_inspected_1103-1106'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_checked_1005-1026_ncrp_vqa_after_SFT')
def dpo_preference_checked_1005_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_human_inspected_1005-1026'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_after_SFT_with_vqa_logp_train*.tsv')


@register_data_path('dpo_checked_1103-1106_ncrp_vqa_after_SFT')
def dpo_preference_checked_1103_1106_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_human_inspected_1103-1106'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_after_SFT_with_vqa_logp_train*.tsv')


@register_data_path('dpo_1122-1123_ncrp_vqa')  # llava vqa
def dpo_preference_1122_1123_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231122to1123-1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1128_ncrp_vqa')  # llava detail
def dpo_preference_1128_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231128to1128'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1122-1123_llava')  # llava vqa
def dpo_preference_1122_1123_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231122to1123-1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_llava_logp_train*.tsv')


@register_data_path('dpo_1128_llava')  # llava detail
def dpo_preference_1128_llava_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231128to1128'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_llava_logp_train*.tsv')


@register_data_processor('dpo_1122-1123_llava')  # llava vqa
def dpo_1122_1123_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1128_llava')  # llava detail
def dpo_1128_llava_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)

# @register_data_path('dpo_1107_ncrp_vqa')
# def dpo_preference_1107_no_crop_vqa_path():
#     data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231107'
#     return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_gpt4v_expansion_ncrp_vqa')
def dpo_preference_gpt4v_expansion_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_gpt4v_preference_humanann_20231108to1108'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1005-1026_g_4v1103')
def dpo_preference_1005_1026_g_4v1103_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1005-1026_good'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1103_logp_train*.tsv')


@register_data_path('dpo_1027-1101-2_g_4v1103')
def dpo_preference_1027_1101_g_4v1103_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231027to1101-2'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1103_logp_train*.tsv')


@register_data_path('dpo_gpt4v_1005-1101-2_g_4v1103_gpt4_1107_with_VQA')
def dpo_gpt4v_preference_1005_1101_g_4v1103_withvqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_gpt4v_preference_20231005to1101-2'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1107wvqa_logp_train*.tsv')


@register_data_path('dpo_gpt4v_1005-1101-2_g_4v1103_gpt4_1107')
def dpo_gpt4v_preference_1005_1101_g_4v1103_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_gpt4v_preference_20231005to1101-2'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1107_logp_train*.tsv')


@register_data_path('dpo_1027-1101-2_g_4v1103_gpt4_1107_with_VQA')
def dpo_preference_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231027to1101-2'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1107wvqa_logp_train*.tsv')


@register_data_processor('dpo_1027-1101-2_g_4v1103_gpt4_1107_with_VQA')
def dpo_preference_1012_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1005to1026_good_gpt4_1107_with_VQA')
def dpo_preference_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1005-1026_good'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1107wvqa_logp_train*.tsv')


@register_data_processor('dpo_1005to1026_good_gpt4_1107_with_VQA')
def dpo_preference_1012_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1027-1101-2_g_4v1103_gpt4_1107')
def dpo_preference_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231027to1101-2'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1107_logp_train*.tsv')


@register_data_processor('dpo_1027-1101-2_g_4v1103_gpt4_1107')
def dpo_preference_1012_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1005to1026_good_gpt4_1107')
def dpo_preference_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1005-1026_good'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1107_logp_train*.tsv')


@register_data_processor('dpo_1005to1026_good_gpt4_1107')
def dpo_preference_1012_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1005to1026_good_gpt4_1103')
def dpo_preference_1026_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1005-1026_good'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_gpt4v1103_logp_train*.tsv')


@register_data_processor('dpo_1005to1026_good_gpt4_1103')
def dpo_preference_1012_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_1023_ncrp_vqa_ranking')
def dpo_1023_ncrp_vqa_ranking_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231023_ranking'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1024_ncrp_vqa_ranking')
def dpo_1024_ncrp_vqa_ranking_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231024_ranking'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1025_ncrp_vqa_ranking')
def dpo_preference_1025_no_crop_vqa_ranking_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231025-new_ranking'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_1026_ncrp_vqa_ranking')
def dpo_preference_1026_no_crop_vqa_ranking_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231026-new_ranking'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1013_add1_no_crop_vqa')
def dpo_preference_1013_add1_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231013_add1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1016_add1_no_crop_vqa')
def dpo_preference_1016_add1_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231016_add1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1017_add1_no_crop_vqa')
def dpo_preference_1017_add1_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231017_add1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1018_add1_no_crop_vqa')
def dpo_preference_1018_add1_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231018_add1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_1019_add1_no_crop_vqa')
def dpo_preference_1019_add1_no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231019_add1'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_preference_gpt4v-nocrop-merge')
def dpo_preference_gpt4v_nocrop_merge__no_crop_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_gpt4v-nocrop_merge'
    return gather_data_files_by_glob(data_dir, pattern='*with_per_token_vqa_logp_train*.tsv')


@register_data_processor('dpo_ncrp_vqa_1005')
def dpo_preference_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1012_no_crop_vqa')
def dpo_preference_1012_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1013_no_crop_vqa')
def dpo_preference_1013_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1016_ncrp_vqa')
def dpo_preference_1016_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1017_no_crop_vqa')
def dpo_preference_1017_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1018_no_crop_vqa')
def dpo_preference_1018_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1019_ncrp_vqa')
def dpo_preference_1019_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1020_no_crop_vqa')
def dpo_preference_1020_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1023_ncrp_vqa')
def dpo_1023_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1024_ncrp_vqa')
def dpo_1024_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1025_ncrp_vqa')
def dpo_1025_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1026_ncrp_vqa')
def dpo_1026_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1005-1026_ncrp_vqa')
def dpo_1005_1026_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1027-1101-2_ncrp_vqa')
def dpo_1027_1101_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1103-1106_ncrp_vqa')
def dpo_1103_1106_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1122-1123_ncrp_vqa')  # llava vqa
def dpo_1122_1123_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1128_ncrp_vqa')  # llava detail
def dpo_1128_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_checked_1005-1026_ncrp_vqa')
def dpo_checked_1005_1026_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_checked_1103-1106_ncrp_vqa')
def dpo_checked_1103_1106_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_checked_1005-1026_ncrp_vqa_after_SFT')
def dpo_checked_1005_1026_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_checked_1103-1106_ncrp_vqa_after_SFT')
def dpo_checked_1103_1106_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1005-1026_g_4v1103')
def dpo_1005_1026_g_4v1103_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1027-1101-2_g_4v1103')
def dpo_1027_1101_g_4v1103_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_gpt4v_expansion_ncrp_vqa')
def dpo_gpt4v_expansion_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_gpt4v_1005-1101-2_g_4v1103_gpt4_1107_with_VQA')
def dpo_gpt4_1005_1101_g_4v1103_gpt4_1107_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_gpt4v_1005-1101-2_g_4v1103_gpt4_1107')
def dpo_gpt4_1005_1101_g_4v1103_gpt4_1107_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1023_ncrp_vqa_ranking')
def dpo_1023_ncrp_vqa_ranking_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1024_ncrp_vqa_ranking')
def dpo_1024_ncrp_vqa_ranking_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1025_ncrp_vqa_ranking')
def dpo_1025_ncrp_vqa_ranking_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1026_ncrp_vqa_ranking')
def dpo_1026_ncrp_vqa_ranking_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1013_add1_no_crop_vqa')
def dpo_preference_1013_add1_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1016_add1_no_crop_vqa')
def dpo_preference_1016_add1_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1017_add1_no_crop_vqa')
def dpo_preference_1017_add1_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1018_add1_no_crop_vqa')
def dpo_preference_1018_add1_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1019_add1_no_crop_vqa')
def dpo_preference_1019_add1_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_llava')
def dpo_preference_llava_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_1012_llava')
def dpo_preference_1012_llava_no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_preference_gpt4v-nocrop-merge')
def dpo_preference_gpt4v_nocrop_merge__no_crop_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_processor('dpo_1005')
def dpo_preference_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                             intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft' or intent == 'eval':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        assert len(list_conv) in [
            3, 4], f'length must be in [3, 4] for data w/ or w/o logps, bug got {len(list_conv)}'

        question = list_conv[0]
        if '<image>' not in question:
            question = f"<image>\n{question}"

        out_chosen = list_conv[1]
        out_rejected = list_conv[2]

        question = {"from": "human", "value": question}
        out_chosen = {"from": "gpt", "value": out_chosen}
        out_rejected = {"from": "gpt", "value": out_rejected}

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # dpo data dir
            "origin_split": origin_split,  # dpo metainfo
            "origin_idx": origin_split_inner_idx,  # index in dpo parquet file
            "image_id": img_path,  # cocoid
        }

        data_dict = {
            'image': image,
            'question': question,
            'chosen': out_chosen,
            'rejected': out_rejected,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }

        if len(list_conv) == 4:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
             data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = list_conv[3]

        return data_dict
    else:
        raise NotImplemented


@register_data_processor('dpo_preference_1012')
def dpo_preference_1012_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


def dpo_preference_llava_rlhf_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        assert len(list_conv) in [
            3, 4], f'length must be in [3, 4] for data w/ or w/o logps, bug got {len(list_conv)}'

        # question type should be [{"from": "", "value": ""}]
        question = list_conv[0]
        assert type(question) == type([])
        assert "value" in question[0].keys()
        assert "from" in question[0].keys()

        for i, item in enumerate(question):
            if i % 2 == 0:
                question[i]["from"] = "human"
            else:
                question[i]["from"] = "gpt"

            if i == 0:
                if '<image>' not in question[i]["value"]:
                    question[i]["value"] = f"<image>\n{question[i]['value']}"
                else:
                    question[i]["value"] = question[i]["value"].replace(
                        "<image>", "").strip()
                    assert "\n" not in question
                    question[i]["value"] = f"<image>\n{question[i]['value']}"

        out_chosen = list_conv[1]
        out_rejected = list_conv[2]

        out_chosen = {"from": "gpt", "value": out_chosen}
        out_rejected = {"from": "gpt", "value": out_rejected}

        out_chosen_conv = copy.deepcopy(question)
        out_chosen_conv.append(out_chosen)
        out_rejected_conv = copy.deepcopy(question)
        out_rejected_conv.append(out_rejected)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # dpo data dir
            "origin_split": origin_split,  # dpo metainfo
            "origin_idx": origin_split_inner_idx,  # index in dpo parquet file
            "image_id": img_path,  # cocoid
        }

        data_dict = {
            'image': image,
            'question': "",
            'chosen': out_chosen_conv,
            'rejected': out_rejected_conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }

        if len(list_conv) == 4:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
             data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = list_conv[3]

        return data_dict
    else:
        raise NotImplemented


@register_data_processor('dpo_llavarlhf_hallonly_ncrp_vqa')
def dpo_llavarlhf_hallonly_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_processor('dpo_llavarlhf_hallonly_ncrp_vqa_238')
def dpo_llavarlhf_hallonly_238_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_llavarlhf_hallonly_ncrp_vqa')
def dpo_llavarlhf_hallonly_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_hallonly'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_vqa_logp_train*.tsv')


@register_data_path('dpo_llavarlhf_hallonly_ncrp_vqa_238')
def dpo_llavarlhf_hallonly_238_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_hallonly_238'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_vqa_logp_train*.tsv')


@register_data_processor('dpo_llavarlhf_nohall_ncrp_vqa')
def dpo_llavarlhf_nohall_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('dpo_llavarlhf_nohall_ncrp_vqa')
def dpo_llavarlhf_nohall_ncrp_vqa_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_nothall'
    return gather_data_files_by_glob(data_dir, pattern='*dpo_with_per_token_vqa_logp_train*.tsv')


@register_data_processor('dpo_preference_llava_7b_v1_preference_hallonly')
def dpo_preference_llava_rlhf_hallonly_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_processor('dpo_preference_llava_7b_v1_preference_hallonly_nocontext_no_crop_vqa')
def dpo_preference_llava_rlhf_hallonly_nocontext_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_processor('dpo_preference_llava_7b_v1_preference_nothall')
def dpo_preference_llava_rlhf_nothall_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_processor('dpo_preference_llava_7b_v1_preference_nothall_nocontext_no_crop_vqa')
def dpo_preference_llava_rlhf_nothall_nocontext_processor(*args, **kwargs):
    return dpo_preference_llava_rlhf_processor(*args, **kwargs)


@register_data_path('CoH_1005')
def coh_preference_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    return gather_data_files_by_glob(data_dir, pattern='*with_logp_train*.tsv')


@register_data_processor('CoH_1005')
def coh_preference_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                             intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        assert len(list_conv) in [
            3, 4], f'length must be in [3, 4] for data w/ or w/o logps, bug got {len(list_conv)}'

        response_chosen = list_conv[1]
        response_rejected = list_conv[2]

        coh_response_A = f'The following is a response with hallucination. {response_rejected}\nThe following is a response without hallucination. {response_chosen}'
        coh_response_B = f'The following is a response without hallucination. {response_chosen}\nThe following is a response with hallucination. {response_rejected}'
        coh_response_C = f'Generate a response without errors. {response_chosen}\nGenerate a response with errors. {response_rejected}'
        coh_response_D = f'Generate a response with errors. {response_rejected}\nGenerate a response without errors. {response_chosen}'
        coh_response = random.choice(
            [coh_response_A, coh_response_B, coh_response_C, coh_response_D])
        question, out = wrap_qa_to_single_turn_multimodal_conv(
            coh_response, list_conv[0])

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # dpo data dir
            "origin_split": origin_split,  # dpo metainfo
            "origin_idx": origin_split_inner_idx,  # index in dpo parquet file
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('dpo_preference_val')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    return gather_data_files_by_glob(data_dir, pattern='omnilmm_long_rewrite_1005_good_val_dpo_with_per_token_vqa_logp_val-32.tsv')


@register_data_processor('dpo_preference_val')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_qwen_long_norewrite')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231220_3to1221_2'
    return gather_data_files_by_glob(data_dir, pattern='dpo_preference_1220_3to1221_2_dpo_with_per_token_vqa_logp_train-56.tsv')


@register_data_processor('dpo_val_qwen_long_norewrite')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_qwen_long_mianbi')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231220_3to1221_2'
    return gather_data_files_by_glob(data_dir, pattern='qwen_long_rewrite_bymianbi_1220_1221_val_dpo_with_per_token_vqa_logp_train-48.tsv')


@register_data_processor('dpo_val_qwen_long_mianbi')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_qwen_long')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231220_3to1221_2'
    return gather_data_files_by_glob(data_dir, pattern='qwen_long_rewrite_1220_1221_val_dpo_with_per_token_vqa_logp_train-48.tsv')


@register_data_processor('dpo_val_qwen_long')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_omnilmm_long')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231113-1to1116'
    return gather_data_files_by_glob(data_dir, pattern='omnilmm_long_rewrite_1113-1to1116_good_val_dpo_with_per_token_vqa_logp_val-48.tsv')


@register_data_processor('dpo_val_omnilmm_long')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_llava_long')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231130to1204'
    return gather_data_files_by_glob(data_dir, pattern='llava_long_rewrite_1130to1204_good_val_dpo_with_per_token_vqa_logp_val-48.tsv')


@register_data_processor('dpo_val_llava_long')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_omnilmm_short')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231107to1110-2_goodqas'
    return gather_data_files_by_glob(data_dir, pattern='omnilmm_short_rewrite_1107to1110-2_good_val_dpo_with_per_token_vqa_logp_val-48.tsv')


@register_data_processor('dpo_val_omnilmm_short')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


@register_data_path('dpo_val_llava_short')
def dpo_preference_data_val_path():
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231121-2to1121-2'
    return gather_data_files_by_glob(data_dir, pattern='llava_short_rewrite_1121-2_good_val_dpo_with_per_token_vqa_logp_val-48.tsv')


@register_data_processor('dpo_val_llava_short')
def dpo_preference_val_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


def dpo_preference_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                 intent, img_transformer=None):
    if intent == 'eval':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        assert len(list_conv) in [
            3, 4], f'length must be in [3, 4] for data w/ or w/o logps, bug got {len(list_conv)}'

        question = list_conv[0]
        if '<image>' not in question:
            question = f"<image>\n{question}"

        out_chosen = list_conv[1]
        out_rejected = list_conv[2]

        question = {"from": "human", "value": question}
        out_chosen = {"from": "gpt", "value": out_chosen}
        out_rejected = {"from": "gpt", "value": out_rejected}

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # dpo data dir
            "origin_split": origin_split,  # dpo metainfo
            "origin_idx": origin_split_inner_idx,  # index in dpo parquet file
            "image_id": img_path,  # cocoid
        }

        data_dict = {
            'image': image,
            'question': question,
            'chosen': out_chosen,
            'rejected': out_rejected,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }

        if len(list_conv) == 4:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'],
             data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp']) = list_conv[3]

        return data_dict
    else:
        raise NotImplemented


def description_instruction_templates():
    # instructions = [
    #     "Describe the following image in detail.",
    #     "Provide a detailed description of the given image.",
    #     "Give an elaborate explanation of the image you see.",
    #     "Share a comprehensive rundown of the presented image.",
    #     "Offer a thorough analysis of the image.",
    #     "Explain the various aspects of the image before you.",
    #     "Clarify the contents of the displayed image with great detail.",
    #     "Characterize the image using a well-detailed description.",
    #     "Break down the elements of the image in a detailed manner.",
    #     "Walk through the important details of the image.",
    #     "Portray the image with a rich, descriptive narrative.",
    #     "Narrate the contents of the image with precision.",
    #     "Analyze the image in a comprehensive and detailed manner.",
    #     "Illustrate the image through a descriptive explanation.",
    #     "Examine the image closely and share its details.",
    #     "Write an exhaustive depiction of the given image.",
    # ]

    # instructions = [
    #     "Identify and describe each object in the image in detail.",
    #     "Describe the key features of the image in great detail.",
    #     "What are the main elements in this image? Describe them thoroughly.",
    #     "Explain what's happening in the image with as much detail as possible.",
    #     "Detail the image's components with particular focus on each entity.",
    #     "Provide an intricate description of every entity in the image.",
    #     "What are the main objects or subjects in the image? Please describe them in detail.",
    #     "What is the setting or environment in which the image takes place?",
    #     "How do the elements in the image relate to each other in terms of positioning or composition?",
    #     "Explain the elements of the image with thorough attention to detail.",
    #     "Explain the image's various components in depth.",
    #     "What are the key features you observe in the image?",
    #     "Can you point out the details that make this image unique?",
    #     "Itemize the elements you identify in the image and describe them thoroughly.",
    #     "Convey the specifics of the image with meticulous attention to detail.",
    #     "Tell me what catches your eye in the image, and describe those elements in depth.",
    # ]

    instructions = [
        "Describe the following image.",
        "Provide a description of the given image."
    ]

    question = random.choice(instructions)

    return question


@register_data_path('coco-detail')
def coco_detail_train_data_path():
    data_dir = '/home/zhanghaoye/data/tsv_files/COCODetail/'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('coco-detail')
def coco_detail_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                intent, img_transformer=None):
    if intent == 'eval':
        origin_qa = description_instruction_templates()

        out: List[dict] = []
        out = []

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # coco-image
            "origin_split": origin_split,  # coco-image split: train
            "origin_idx": origin_split_inner_idx,  # cocoid
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
            'origin_question': origin_qa,
        }
    else:
        raise NotImplemented


@register_data_path('okvqa-train')
def okvqa_train_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/okvqa'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'train' in f]
    return data_dir, filenames


@register_data_processor('okvqa-train')
def okvqa_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        question = vqa_instruction_templates(
            question)  # add short answer instruction

        answer = origin_qa["answer"]
        org_answers = origin_qa["org_answers"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # okvqa
            "origin_split": origin_split,  # okvqa split: val
            "origin_idx": origin_split_inner_idx,  # question_id in okvqa
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'gt_answers': org_answers,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    elif intent == 'eval':
        return okvqa_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                   intent, img_transformer=None)
    else:
        raise NotImplemented


@register_data_path('okvqa-val')
def okvqa_val_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/okvqa'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'val' in f]
    return data_dir, filenames


@register_data_processor('okvqa-val')
def okvqa_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'eval':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]
        org_answers = origin_qa["org_answers"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # okvqa
            "origin_split": origin_split,  # okvqa split: val
            "origin_idx": origin_split_inner_idx,  # question_id in okvqa
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'gt_answers': org_answers,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
            'origin_question': origin_qa["question"]
        }
    else:
        raise NotImplemented


@register_data_path('vqav2-val')
def vqav2_val_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/VQAv2/vqav2_full_tsv'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'val' in f]
    return data_dir, filenames


@register_data_path('vqav2-train-wrong')
def vqav2_train_wrong_data_path():
    data_dir = '/home/zhanghaoye/data/tsv_files/vqav2_ncrp_train_wrong'
    _, filenames = gather_data_files_by_glob(data_dir)
    return data_dir, filenames


@register_data_processor('vqav2-train-wrong')
def vqav2_train_wrong_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                intent, img_transformer=None):
    if intent == 'eval':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # vqav2
            "origin_split": origin_split,  # vqav2 split: val
            "origin_idx": int(origin_split_inner_idx),  # question_id in vqav2
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
            'origin_question': origin_qa["question"],
            "gt_answers": answer,
        }
    else:
        raise NotImplemented


@register_data_processor('vqav2-val')
def vqav2_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'eval':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # vqav2
            "origin_split": origin_split,  # vqav2 split: val
            "origin_idx": int(origin_split_inner_idx),  # question_id in vqav2
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
            'origin_question': origin_qa["question"],
        }
    else:
        raise NotImplemented


@register_data_path('vqav2-train')
def vqav2_train_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/VQAv2/vqav2_full_tsv'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'train' in f]
    return data_dir, filenames


@register_data_processor('vqav2-train')
def vqav2_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]
        question = vqa_instruction_templates(
            question)  # vqa short answer template

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # vqav2
            "origin_split": origin_split,  # vqav2 split: train
            "origin_idx": origin_split_inner_idx,  # question_id in vqav2
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    elif intent == 'eval':
        return vqav2_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                   intent, img_transformer)
    else:
        raise NotImplemented


@register_data_path('aokvqa-val')
def aokvqa_val_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/aokvqa'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'val' in f]
    return data_dir, filenames


@register_data_processor('aokvqa-val')
def aokvqa_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                         intent, img_transformer=None):
    if intent == 'eval':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]
        direct_answer = origin_qa["direct_answer"]
        difficult_direct_answer = origin_qa["difficult_direct_answer"]

        rationales = origin_qa["rationales"]
        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: val
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('aokvqa-train')
def aokvqa_train_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/a-okvqa'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'train' in f]
    return data_dir, filenames


@register_data_processor('aokvqa-train')
def aokvqa_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                           intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        question = vqa_instruction_templates(question)

        answer = origin_qa["answer"]
        direct_answer = origin_qa["direct_answer"]
        difficult_direct_answer = origin_qa["difficult_direct_answer"]

        rationales = origin_qa["rationales"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('TextVQA')
def textvqa_train_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/TextVQA'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('TextVQA')
def textvqa_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                            intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        question = textvqa_instruction_templates(question)

        answer = origin_qa["answers"]
        from collections import Counter
        answer_count = Counter(answer)
        answer_keep = [key for key, item in answer_count.items() if item >= 3]
        if len(answer_keep) == 0:
            answer_keep = answer
        answer = random.choice(answer_keep)

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('DocVQA')
def docvqa_train_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/DocVQA'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('DocVQA')
def docvqa_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                           intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        question = textvqa_instruction_templates(question)

        answer = origin_qa["answers"]
        answer = random.choice(answer)

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('OCRVQA')
def ocrvqa_train_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/OCRVQA'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('OCRVQA')
def ocrvqa_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                           intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        question = textvqa_instruction_templates(question)

        answer = origin_qa["answers"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(
            answer, question)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('synthdog_en')
def synthdog_en_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/synthdog'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('synthdog_en')
def synthdog_en_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('laion_400m')
def laion_400m_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/laion_aesthetics_v2_5plus_400m_0328/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_path('laion_2b_1')
def laion_2b_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/laion_2b_20231201_export/tsv/2023-12-01'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('laion_2b_2')
def laion_2b_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/laion_2b_20231201_export/tsv-2/2023-12-01'
    return gather_data_files_by_glob(data_dir, '*.tsv')


def register_shards(name, root_pattern, file_pattern, depth=1):
    for i in range(10):
        ds_name = f'{name}_{i}'

        @register_data_path(ds_name)
        def data_path():
            filenames = []
            root = root_pattern
            for fullpath in glob.glob(f'{root}{file_pattern}'):
                filename = '/'.join(fullpath.split('/')[-depth:])
                filenames.append(filename)

            filenames = sorted(filenames)
            index = i
            shard = 10
            shard_size = math.ceil(len(filenames) // shard)
            filenames = filenames[shard_size * index: shard_size * (index + 1)]
            return root, filenames

        @register_data_processor(ds_name)
        def data_processor(*args, **kwargs):
            return laion_400m_processor(*args, **kwargs)


register_shards('laion_2b_clean',
                '/data/public/multimodal/multimodal_data/laion_2b_20231201_export/', 'tsv*/*/*.tsv', 3)
register_shards('laion_coco_clean',
                '/data/public/multimodal/multimodal_data/laion_coco_20231201_export/', 'tsv*/*.tsv', 2)
register_shards(
    'coyo_clean', '/data/public/multimodal/multimodal_data/coyo_20231201_export/tsv/', '*.tsv')

register_shards(
    'l2b_c', '/data/public/multimodal/multimodal_data/laion_2b_20231201_export/', 'tsv*/*/*.tsv', 3)
register_shards(
    'lcc_c', '/data/public/multimodal/multimodal_data/laion_coco_20231201_export/', 'tsv*/*.tsv', 2)
register_shards(
    'cy_c', '/data/public/multimodal/multimodal_data/coyo_20231201_export/tsv/', '*.tsv')


@register_data_processor(['laion_400m', 'laion_coco', 'laion_2b_1', 'laion_2b_2'])
def laion_400m_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                         intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)['text']
        conv = wrap_caption_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  #
            # metainfos, [file_name, data_type, uid, clip-score] for laion_2b, file_name for others
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,  # line index in parquet
            "image_id": img_path,  # image_id
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('laion_coco')
def laion_coco_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/laion_coco/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_path('pretrain_eval_train')
def pretrain_eval_train_data_path():
    data_dir = '/data/public/multimodal/pretrain_eval_tsv/train/'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('pretrain_eval_eval')
def pretrain_eval_train_data_path():
    data_dir = '/data/public/multimodal/pretrain_eval_tsv/eval/'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_path('cc12m')
def cc12m_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/cc12m_split_20230331150504/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('cc12m')
def cc12m_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                    intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)[0]
        clean = out.replace('<PERSON>', '')
        if len(clean.split()) < 3:
            clean = out.replace('<PERSON>', 'person')

        conv = wrap_caption_generation_single_turn_conv(clean)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('commoncrawl')
def commoncrawl_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/CommonCrawl_HTML/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('commoncrawl')
def commoncrawl_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)['text']
        conv = wrap_ocr_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('c4web')
def c4web_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/c4web/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('c4web')
def c4web_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                    intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)['text']
        conv = wrap_ocr_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


def reverse_cc_text(text):
    import re
    box_pattern = re.compile(r'<quad>(.*?)</quad>', re.DOTALL)
    text_pattern = re.compile(r'<ref>(.*?)</ref>', re.DOTALL)
    boxes = box_pattern.findall(text)
    texts = text_pattern.findall(text)
    text = ''.join(
        [f'<ref>{text}</ref><quad>{box}</quad>' for box, text in zip(boxes, texts)])
    return text


@register_data_path('commoncrawl_reverse')
def commoncrawl_reverse_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/CommonCrawl_HTML/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('commoncrawl_reverse')
def commoncrawl_reverse_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                  intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)['text']
        out = reverse_cc_text(out)
        conv = wrap_ocr_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('c4web_reverse')
def c4web_reverse_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/c4web/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('c4web_reverse')
def c4web_reverse_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                            intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)['text']
        out = reverse_cc_text(out)
        conv = wrap_ocr_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('coco')
def coco_data_path():
    data_dir = '/data/public/multimodal/multimodal_data/coco/tsv'
    return gather_data_files_by_glob(data_dir)


@register_data_processor('coco')
def coco_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                   intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = "".join(json.loads(text))
        out = random.choice(out.split('<cap_sep>'))
        conv = wrap_caption_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('vg')
def vg_data_path():
    data_dir = "/data/public/multimodal/multimodal_data/VisualGenome/tsv"
    return gather_data_files_by_glob(data_dir)


@register_data_processor('vg')
def vg_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                 intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)
        out = random.choice(''.join(out).split('<cap_sep>'))
        conv = wrap_caption_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_path('cc3m')
def cc3m_data_path():
    data_dir = "/data/public/multimodal/multimodal_data/CC3M_tsv/tsv"
    return gather_data_files_by_glob(data_dir)


@register_data_processor('cc3m')
def cc3m_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                   intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        out = json.loads(text)
        out = ''.join(out)
        conv = wrap_caption_generation_single_turn_conv(out)
        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,  # aokvqa
            "origin_split": origin_split,  # aokvqa split: train
            "origin_idx": origin_split_inner_idx,  # line index in aokvqa_val json
            "image_id": img_path,  # cocoid
        }

        return {
            'image': image,
            'conversations': conv,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented
