# flake8: noqa: F401, F403
import abc
import argparse
import csv
import multiprocessing as mp
import os
import os.path as osp
import copy as cp
import random as rd
import requests
import shutil
import subprocess
import warnings
import logging
import pandas as pd
from collections import OrderedDict, defaultdict
from multiprocessing import Pool, current_process
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
from json import JSONDecoder
from huggingface_hub import scan_cache_dir
from sty import fg, bg, ef, rs

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def h2r(value):
    if value[0] == '#':
        value = value[1:]
    assert len(value) == 6
    return tuple(int(value[i:i + 2], 16) for i in range(0, 6, 2))

def r2h(rgb):
    return '#%02x%02x%02x' % rgb

def colored(s, color):
    if isinstance(color, str):
        if hasattr(fg, color):
            return getattr(fg, color) + s + fg.rs
        color = h2r(color)
    return fg(*color) + s + fg.rs

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False

def bincount(lst):
    bins = defaultdict(lambda: 0)
    for item in lst:
        bins[item] += 1
    return bins

def get_cache_path(repo_id, branch=None):
    hf_cache_info = scan_cache_dir()
    repos = list(hf_cache_info.repos)
    repo = None
    for r in repos:
        if r.repo_id == repo_id:
            repo = r
            break
    if repo is None:
        return None
    revs = list(repo.revisions)
    if branch is not None:
        revs = [r for r in revs if r.refs == frozenset({branch})]
    rev2keep, last_modified = None, 0
    for rev in revs:
        if rev.last_modified > last_modified:
            rev2keep, last_modified = rev, rev.last_modified
    if rev2keep is None:
        return None
    return str(rev2keep.snapshot_path)

def proxy_set(s):
    import os
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def splitlen(s, sym='/'):
    return len(s.split(sym))

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def d2df(D):
    return pd.DataFrame({x: [D[x]] for x in D})

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

try:
    import decord
except ImportError:
    pass

def timestr(second=True, minute=False):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]

def dict_merge(dct, merge_dct):
    for k, _ in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def youtube_dl(idx):
    cmd = f'youtube-dl -f best -f mp4 "{idx}"  -o {idx}.mp4'
    os.system(cmd)

def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd).decode()

def load_env():
    logger = logging.getLogger('LOAD_ENV')
    try:
        import vlmeval
    except ImportError:
        logger.error('VLMEval is not installed. Failed to import environment variables from .env file. ')
        return
    pth = osp.realpath(vlmeval.__path__[0])
    pth = osp.join(pth, '../.env')
    pth = osp.realpath(pth)
    if not osp.exists(pth):
        logger.error(f'Did not detect the .env file at {pth}, failed to load. ')
        return

    from dotenv import dotenv_values
    values = dotenv_values(pth)
    for k, v in values.items():
        if v is not None and len(v):
            os.environ[k] = v
    logger.info(f'API Keys successfully loaded from {pth}')

def pip_install_robust(package):
    import sys
    retry = 3
    while retry > 0:
        try:
            package_base = package.split('=')[0]
            module = __import__(package)
            return True
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            retry -= 1
    return False


def version_cmp(v1, v2, op='eq'):
    from packaging import version
    import operator
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError


def extract_json_objects(text, decoder=JSONDecoder()):
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1: break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1
