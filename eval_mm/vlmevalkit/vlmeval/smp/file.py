import json
import pickle
import pandas as pd
import os
import csv
import hashlib
import os.path as osp
import time
import numpy as np
import validators
import mimetypes
import multiprocessing as mp
from .misc import toliststr
from .vlm import decode_base64_to_image_file


def decode_img_omni(tup):
    root, im, p = tup
    images = toliststr(im)
    paths = toliststr(p)
    if len(images) > 1 and len(paths) == 1:
        paths = [osp.splitext(p)[0] + f'_{i}' + osp.splitext(p)[1] for i in range(len(images))]

    assert len(images) == len(paths)
    paths = [osp.join(root, p) for p in paths]
    for p, im in zip(paths, images):
        if osp.exists(p):
            continue
        if isinstance(im, str) and len(im) > 64:
            decode_base64_to_image_file(im, p)
    return paths


def localize_df(data, dname, nproc=32):
    assert 'image' in data
    indices = list(data['index'])
    indices_str = [str(x) for x in indices]
    images = list(data['image'])
    image_map = {x: y for x, y in zip(indices_str, images)}

    root = LMUDataRoot()
    root = osp.join(root, 'images', dname)
    os.makedirs(root, exist_ok=True)

    if 'image_path' in data:
        img_paths = list(data['image_path'])
    else:
        img_paths = []
        for i in indices_str:
            if len(image_map[i]) <= 64:
                idx = image_map[i]
                assert idx in image_map and len(image_map[idx]) > 64
                img_paths.append(f'{idx}.jpg')
            else:
                img_paths.append(f'{i}.jpg')

    tups = [(root, im, p) for p, im in zip(img_paths, images)]

    pool = mp.Pool(32)
    ret = pool.map(decode_img_omni, tups)
    pool.close()
    data.pop('image')
    if 'image_path' not in data:
        data['image_path'] = [x[0] if len(x) == 1 else x for x in ret]
    return data


def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root


def HFCacheRoot():
    cache_list = ['HUGGINGFACE_HUB_CACHE', 'HF_HOME']
    for cache_name in cache_list:
        if cache_name in os.environ and osp.exists(os.environ[cache_name]):
            if os.environ[cache_name].split('/')[-1] == 'hub':
                return os.environ[cache_name]
            else:
                return osp.join(os.environ[cache_name], 'hub')
    home = osp.expanduser('~')
    root = osp.join(home, '.cache', 'huggingface', 'hub')
    os.makedirs(root, exist_ok=True)
    return root


def MMBenchOfficialServer(dataset_name):
    root = LMUDataRoot()

    if dataset_name in ['MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11']:
        ans_file = f'{root}/{dataset_name}.tsv'
        if osp.exists(ans_file):
            data = load(ans_file)
            if 'answer' in data and sum([pd.isna(x) for x in data['answer']]) == 0:
                return True

    if dataset_name in ['MMBench_TEST_EN', 'MMBench_TEST_CN', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11']:
        ans_file1 = f'{root}/{dataset_name}.tsv'
        mapp = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_CN': 'MMBench_CN',
            'MMBench_TEST_EN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11',
        }
        ans_file2 = f'{root}/{mapp[dataset_name]}.tsv'
        for f in [ans_file1, ans_file2]:
            if osp.exists(f):
                data = load(f)
                if 'answer' in data and sum([pd.isna(x) for x in data['answer']]) == 0:
                    return True
    return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def download_file(url, filename=None):
    import urllib.request
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is None:
        filename = url.split('/')[-1]

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    except Exception as e:
        import logging
        logging.warning(f'{type(e)}: {e}')
        # Handle Failed Downloads from huggingface.co
        if 'huggingface.co' in url:
            url_new = url.replace('huggingface.co', 'hf-mirror.com')
            try:
                download_file(url_new, filename)
                return filename
            except Exception as e:
                logging.warning(f'{type(e)}: {e}')
                raise Exception(f'Failed to download {url}')
        else:
            raise Exception(f'Failed to download {url}')

    return filename


def ls(dirname='.', match=[], mode='all', level=1):
    if isinstance(level, str):
        assert '+' in level
        level = int(level[:-1])
        res = []
        for i in range(1, level + 1):
            res.extend(ls(dirname, match=match, mode='file', level=i))
        return res

    if dirname == '.':
        ans = os.listdir(dirname)
    else:
        ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    assert mode in ['all', 'dir', 'file']
    assert level >= 1 and isinstance(level, int)
    if level == 1:
        if isinstance(match, str):
            match = [match]
        for m in match:
            if len(m) == 0:
                continue
            if m[0] != '!':
                ans = [x for x in ans if m in x]
            else:
                ans = [x for x in ans if m[1:] not in x]
        if mode == 'dir':
            ans = [x for x in ans if osp.isdir(x)]
        elif mode == 'file':
            ans = [x for x in ans if not osp.isdir(x)]
        return ans
    else:
        dirs = [x for x in ans if osp.isdir(x)]
        res = []
        for d in dirs:
            res.extend(ls(d, match=match, mode=mode, level=level - 1))
        return res


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f


def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))


def md5(s):
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())


def last_modified(pth):
    stamp = osp.getmtime(pth)
    m_ti = time.ctime(stamp)
    t_obj = time.strptime(m_ti)
    t = time.strftime('%Y%m%d%H%M%S', t_obj)[2:]
    return t


def parse_file(s):
    if osp.exists(s) and s != '.':
        assert osp.isfile(s)
        suffix = osp.splitext(s)[1].lower()
        mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    elif s.startswith('data:image/'):
        # To be compatible with OPENAI base64 format
        content = s[11:]
        mime = content.split(';')[0]
        content = ';'.join(content.split(';')[1:])
        dname = osp.join(LMUDataRoot(), 'files')
        assert content.startswith('base64,')
        b64 = content[7:]
        os.makedirs(dname, exist_ok=True)
        tgt = osp.join(dname, md5(b64) + '.png')
        decode_base64_to_image_file(b64, tgt)
        return parse_file(tgt)
    elif validators.url(s):
        suffix = osp.splitext(s)[1].lower()
        if suffix in mimetypes.types_map:
            mime = mimetypes.types_map[suffix]
            dname = osp.join(LMUDataRoot(), 'files')
            os.makedirs(dname, exist_ok=True)
            tgt = osp.join(dname, md5(s) + suffix)
            download_file(s, tgt)
            return (mime, tgt)
        else:
            return ('url', s)
    else:
        return (None, s)


def file_size(f, unit='GB'):
    stats = os.stat(f)
    div_map = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }
    return stats.st_size / div_map[unit]


def parquet_to_tsv(file_path):
    data = pd.read_parquet(file_path)
    pth = '/'.join(file_path.split('/')[:-1])
    data_name = file_path.split('/')[-1].split('.')[0]
    data.to_csv(osp.join(pth, f'{data_name}.tsv'), sep='\t', index=False)
