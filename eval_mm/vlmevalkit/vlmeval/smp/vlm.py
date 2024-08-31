import os
import io
import pandas as pd
import numpy as np
import string
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
import sys

Image.MAX_IMAGE_PIXELS = 1e9


def rescale_img(img, tgt=None):
    assert isinstance(tgt, tuple) and -1 in tgt
    w, h = img.size
    if tgt[0] != -1:
        new_w, new_h = tgt[0], int(tgt[0] / w * h)
    elif tgt[1] != -1:
        new_w, new_h = int(tgt[1] / h * w), tgt[1]
    img = img.resize((new_w, new_h))
    return img


def concat_images_vlmeval(images, target_size=-1, mode='h', return_image=False):
    from .file import md5

    ims = [Image.open(im) for im in images]
    if target_size != -1:
        ims = [
            rescale_img(im, (-1, target_size) if mode == 'h' else (target_size, -1))
            for im in ims
        ]

    ws, hs = [x.width for x in ims], [x.height for x in ims]
    if mode == 'h':
        new_w, new_h = sum(ws), max(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i]), 0))
    elif mode == 'v':
        new_w, new_h = max(ws), sum(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i], 0)))
    if return_image:
        return dst
    else:
        _str = '\n'.join(images)
        str_md5 = md5(_str)
        tgt = osp.join('/tmp', str_md5 + '.jpg')
        dst.save(tgt)
        return tgt


def mmqa_display(question, target_size=512):
    question = {k.lower(): v for k, v in question.items()}
    keys = list(question.keys())
    keys = [k for k in keys if k not in ['index', 'image']]

    images = question['image']
    if isinstance(images, str):
        images = [images]

    idx = question.pop('index', 'XXX')
    print(f'INDEX: {idx}')

    for im in images:
        image = decode_base64_to_image(im, target_size=target_size)
        display(image)  # noqa: F821

    for k in keys:
        try:
            if not pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')
        except ValueError:
            if False in pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')


def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret


def encode_image_file_to_base64(image_path, target_size=-1):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)


def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def isimg(s):
    return osp.exists(s) or s.startswith('http')


def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False


def gpt_key_set():
    openai_key = os.environ.get('OPENAI_API_KEY', None)
    return isinstance(openai_key, str) and openai_key.startswith('sk-')


def apiok(wrapper):
    s = wrapper.generate('Hello!')
    return wrapper.fail_msg not in s


def circular_pred(df, extract_func=None):
    if extract_func is None:
        extract_func = lambda x: x  # noqa: E731
    df = df.sort_values('index')
    from vlmeval.utils import can_infer_option

    shift = int(1e6)

    choices = [extract_func(x) for x in df['prediction']]
    pred_map = {i: c for i, c in zip(df['index'], choices)}
    flag_map = {i: True for i in pred_map if i < 1e6}
    valid_map = {i: True for i in pred_map if i < 1e6}
    for i in df['index']:
        if i >= shift and pred_map[i] and pred_map[i - shift]:
            if pred_map[i] not in list(
                string.ascii_uppercase
            ) or pred_map[  # noqa: W504
                i - shift
            ] not in list(
                string.ascii_uppercase
            ):

                valid_map[i % shift] = False
                continue
            if (ord(pred_map[i]) - ord(pred_map[i - shift])) % 4 == 1:
                continue
            else:
                flag_map[i % shift] = False
    flag_map = {k: v for k, v in flag_map.items() if valid_map[k]}
    flags = list(flag_map.values())
    return np.mean(flags)
