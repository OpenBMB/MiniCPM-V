# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import io
import errno
import pandas
import base64

import os.path as op

from PIL import Image
from tqdm import tqdm
from typing import List
from omnilmm.data.tsv_file import TSVFile
from omnilmm.data.tsv_file import LARGEST_TSV_SIZE


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def tsv_writer(value_lines: List[List[str]], tsv_file, sep='\t'):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'

    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'

    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert value_lines is not None

        line_value: List[str]
        for line_value in value_lines:
            assert line_value is not None
            line_value = [v if type(v) != bytes else v.decode(
                'utf-8') for v in line_value]
            v = '{0}\n'.format(sep.join(map(str, line_value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)

    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)


def b64img_ok(b64img):
    try:
        img_io = io.BytesIO(base64.b64decode(b64img))
        img_io.seek(0)
        image = Image.open(img_io).convert('RGB')
    except:
        return False
    return True


def write_line(fp, fpidx, line_value, sep, idx):
    for value in line_value:
        assert isinstance(value, str), f'{type(value)}-{value}'

    v = '{0}\n'.format(sep.join(line_value))
    fp.write(v)
    fpidx.write(str(idx) + '\n')
    idx += len(v)
    return idx


def open_new_file(base_name, ext, counter):
    f_name = f"{base_name}_{counter}"
    return open(f"{f_name}.{ext}", 'w'), f_name


def multimodal_img_tsv_writer(value_lines, tsv_file, sep='\t', text_only=False):
    mkdir(op.dirname(tsv_file))
    # lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'

    tsv_name = op.splitext(tsv_file)[0]

    idx = 0
    num_row = 0
    file_counter = 0

    fp, fp_name = open_new_file(tsv_name, 'tsv.tmp', file_counter)
    fpidx, fpidx_name = open_new_file(tsv_name, 'lineidx.tmp', file_counter)

    assert value_lines is not None

    for dataset_name, img_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path in value_lines:
        if not b64img_ok(img_buffer) and not text_only:
            print(
                f'Image value cannot be interpreted as b64 str of image: {origin_dataset} | {origin_split} | {origin_split_inner_idx}')
            continue
            # raise ValueError('Image value cannot be interpreted as b64 str of image')

        line_value = [dataset_name, img_buffer, text_b64, origin_dataset,
                      origin_split, origin_split_inner_idx, img_path]

        if num_row >= LARGEST_TSV_SIZE:  # LARGEST_TSV_SIZE
            fp.close()
            fpidx.close()
            os.rename(f"{fp_name}.tsv.tmp", f'{fp_name}-{num_row}.tsv')
            os.rename(f"{fpidx_name}.lineidx.tmp",
                      f'{fpidx_name}-{num_row}.lineidx')

            file_counter += 1
            num_row = 0
            idx = 0
            fp, fp_name = open_new_file(tsv_name, 'tsv.tmp', file_counter)
            fpidx, fpidx_name = open_new_file(
                tsv_name, 'lineidx.tmp', file_counter)

        idx = write_line(fp, fpidx, line_value, sep=sep, idx=idx)
        num_row += 1

    fp.close()
    fpidx.close()

    os.rename(f"{fp_name}.tsv.tmp", f'{fp_name}-{num_row}.tsv')
    os.rename(f"{fpidx_name}.lineidx.tmp", f'{fpidx_name}-{num_row}.lineidx')


# prev
def multimodal_img_tsv_writer_prev(value_lines, tsv_file, sep='\t', text_only=False):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'

    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    num_row = 0

    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert value_lines is not None

        for dataset_name, img_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path in value_lines:
            if not b64img_ok(img_buffer) and not text_only:
                raise ValueError(
                    'Image value cannot be interpreted as b64 str of image')

            line_value = [dataset_name, img_buffer, text_b64, origin_dataset,
                          origin_split, origin_split_inner_idx, img_path]
            for value in line_value:
                assert isinstance(value, str), f'{type(value)}-{value}'

            v = '{0}\n'.format(sep.join(line_value))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            num_row += 1
            idx = idx = idx + len(v.encode('utf-8'))

    os.rename(tsv_file_tmp, f'{tsv_file}-{num_row}.tsv')
    os.rename(lineidx_file_tmp, f'{tsv_file}-{num_row}.lineidx')
