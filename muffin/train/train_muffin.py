# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import timm
import torch
import logging
import pathlib
import getpass
import transformers

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from utils.utils import is_main_process, get_rank
from utils.diff_lib import get_diff_ids, color_print_diff_pair, split_into_words
from muffin.train.trainers import MuffinTrainer, MuffinDPOTrainer
from muffin.eval.muffin_inference_logp import preference_collator_fn, concate_pad
from muffin import conversation as conversation_lib
from muffin import LlavaLlamaForCausalLM, Beit3LlavaLlamaForCausalLM
from muffin.model.muffin import interpolate_beit3
from muffin.model.utils import stop_gradient_by_name
from muffin.data.datasets import SingleDataSourceDataset, MultiDataSourceDataset
from muffin.data.data_processors import register_data_path
from muffin.train.train_utils import SFT_collator_fn, IGNORE_INDEX, encode_multimodal_sample, encode_multimodal_preference_sample

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    num_query: int = 64


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = 'unimm-chat'
    data_source_weights: str = '100'
    eval_data_source_names: Optional[str] = field(default=None)

    dpo_beta: float = 0.5
    dpo_token_weight: float = 3.0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_steps: int = field(default=1_000)
    no_randaug: bool = False
    fully_tune: bool = False

    task: str = field(
        default='LM',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    dpo_use_average: bool = False
    dpo_token_weighted: bool = False


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def create_multi_data_source_dataset(data_source_names, data_source_weights):
    ds_list = []
    for name in data_source_names:
        ds = SingleDataSourceDataset(name, *register_data_path[name]())
        ds_list.append(ds)
    ds = MultiDataSourceDataset(ds_list, data_source_weights)
    return ds


class LazySupervisedDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()

        logging.warning("Loading data...")
        list_data_dict = create_multi_data_source_dataset(
            multimodal_cfg['data_source_names'], multimodal_cfg['data_source_weights'])

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source: dict = self.list_data_dict[i]
        # print(f'{i} Rank-{get_rank()} {source["idx"]}')

        data_dict = encode_multimodal_sample(
            source, self.tokenizer, self.multimodal_cfg)
        return data_dict


class DPODataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(DPODataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = create_multi_data_source_dataset(
            multimodal_cfg['data_source_names'], multimodal_cfg['data_source_weights'])
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            source, self.tokenizer, self.multimodal_cfg)
        return rej_data_dict, win_data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return SFT_collator_fn(instances, self.tokenizer.pad_token_id)


@dataclass
class DataCollatorForDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta: float
    mod_token_weight: float

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # torch.set_printoptions(linewidth=200)
        batch = preference_collator_fn(instances, self.tokenizer.pad_token_id)

        rej_instances, win_instances = list(zip(*instances))

        batch['beta'] = self.beta
        batch['ref_win_logp'] = torch.as_tensor(
            [x['ref_win_logp'] for x in win_instances])
        batch['ref_rej_logp'] = torch.as_tensor(
            [x['ref_rej_logp'] for x in rej_instances])
        batch['ref_win_avg_logp'] = torch.as_tensor(
            [x['ref_win_avg_logp'] for x in win_instances])
        batch['ref_rej_avg_logp'] = torch.as_tensor(
            [x['ref_rej_avg_logp'] for x in rej_instances])

        ref_win_per_token_logp = [torch.as_tensor(
            x['ref_win_per_token_logp']) for x in win_instances]
        ref_rej_per_token_logp = [torch.as_tensor(
            x['ref_rej_per_token_logp']) for x in rej_instances]

        batch['ref_win_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(
            ref_win_per_token_logp, batch_first=True, padding_value=0)
        batch['ref_rej_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(
            ref_rej_per_token_logp, batch_first=True, padding_value=0)

        win_input_ids = batch['win_input_ids']
        rej_input_ids = batch['rej_input_ids']
        win_labels = batch['win_labels']
        rej_labels = batch['rej_labels']
        assert batch['ref_win_per_token_logp'].size(1) >= win_input_ids.size(
            1) - 1, f"{batch['ref_win_per_token_logp'].size(1)} >= {win_input_ids.size(1) - 1}"
        assert batch['ref_rej_per_token_logp'].size(1) >= rej_input_ids.size(
            1) - 1, f"{batch['ref_rej_per_token_logp'].size(1)} >= {rej_input_ids.size(1) - 1}"

        # length of logp is one-token shorter since the last token's output is not used
        batch['ref_win_per_token_logp'] = batch['ref_win_per_token_logp'][:,
                                                                          :win_input_ids.size(1) - 1]
        batch['ref_rej_per_token_logp'] = batch['ref_rej_per_token_logp'][:,
                                                                          :rej_input_ids.size(1) - 1]

        win_token_weight = torch.ones_like(batch['ref_win_per_token_logp'])
        rej_token_weight = torch.ones_like(batch['ref_rej_per_token_logp'])

        for idx, (w, r, wl, rl, wlogp, rlogp) in enumerate(zip(win_input_ids, rej_input_ids, win_labels, rej_labels, ref_win_per_token_logp, ref_rej_per_token_logp)):
            valid_w = w[1:]
            valid_r = r[1:]
            # print(idx, f'logp shape w={wlogp.shape}, r={rlogp.shape}', flush=True)
            # print(idx, f'valid_w, {len(valid_w)}', self.tokenizer.decode(valid_w), '\n', flush=True)
            # print(idx, f'valid_r, {len(valid_r)}', self.tokenizer.decode(valid_r), '\n', flush=True)

            # print(idx, f'w, {len(w)}', self.tokenizer.decode(w), '\n', flush=True)
            # print(idx, f'r, {len(r)}', self.tokenizer.decode(r), '\n', flush=True)
            # print()

            min_match_size = 3
            # TODO: add junk condition for space tokens like 13 for '\n'
            r_mod, w_mod = get_diff_ids(
                valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            r_mod_tokens = valid_r[r_mod]
            w_mod_tokens = valid_w[w_mod]

            # print(idx, f'mod_w, {self.mod_token_weight} {len(w_mod_tokens)}', self.tokenizer.decode(w_mod_tokens), '\n', flush=True)
            # print(idx, f'mod_r, {self.mod_token_weight} {len(r_mod_tokens)}', self.tokenizer.decode(r_mod_tokens), '\n', flush=True)
            # color_print_diff_pair(split_into_words(self.tokenizer.decode(valid_r)),
            #                       split_into_words(self.tokenizer.decode(valid_w)),
            #                       min_match_size=min_match_size)
            # color_print_diff_pair(valid_r.tolist(),
            #                       valid_w.tolist(),
            #                       min_match_size=min_match_size)

            win_token_weight[idx][w_mod] = self.mod_token_weight
            rej_token_weight[idx][r_mod] = self.mod_token_weight

        batch['win_token_weight'] = win_token_weight
        batch['rej_token_weight'] = rej_token_weight
        batch['concatenated_token_weight'] = concate_pad(
            win_token_weight, rej_token_weight, 0)

        for ins in win_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
            # assert len(ins['input_ids']) == len(ins['ref_win_per_token_logp']), f"{len(ins['input_ids'])} == {len(ins['ref_win_per_token_logp'])}"
            # print('win', len(ins['input_ids']), len(ins['labels']), len(ins['ref_win_per_token_logp']), batch['win_input_ids'].size(1), batch['win_labels'].size(1), batch['concatenated_token_weight'].size(1), flush=True)
        for ins in rej_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
            # assert len(ins['input_ids']) == len(ins['ref_rej_per_token_logp']), f"{len(ins['input_ids'])} == {len(ins['ref_rej_per_token_logp'])}"
            # print('rej', len(ins['input_ids']), len(ins['labels']), len(ins['ref_rej_per_token_logp']), batch['rej_input_ids'].size(1), batch['rej_labels'].size(1), batch['concatenated_token_weight'].size(1), flush=True)
        if torch.any(torch.isnan(batch['win_token_weight'])):
            print(f'win_token_weight fail', flush=True)
            exit()
        if torch.any(torch.isnan(batch['rej_token_weight'])):
            print(f'rej_token_weight fail', flush=True)
            exit()

        # print('win ids', win_input_ids, flush=True)
        # print('rej ids', rej_input_ids, flush=True)
        # print('cat ids', batch['concatenated_input_ids'].tolist(), flush=True)
        # print('win labels', win_labels, flush=True)
        # print('rej labels', rej_labels, flush=True)
        # print('cat labels', batch['concatenated_labels'].tolist(), flush=True)

        # print('win weight', win_token_weight, flush=True)
        # print('rej weight', rej_token_weight, flush=True)
        # print('cat weight', batch['concatenated_token_weight'].tolist(), flush=True)

        # win_weighted_mask = (win_labels[:, 1:].clone() != -100) * win_token_weight
        # rej_weighted_mask = (rej_labels[:, 1:].clone() != -100) * rej_token_weight
        # cat_weighted_mask = (batch['concatenated_labels'][:, 1:].clone() != -100) * batch['concatenated_token_weight']
        # print('win weighted mask', win_weighted_mask, flush=True)
        # print('rej weighted mask', rej_weighted_mask, flush=True)
        # print('cat weighted mask', cat_weighted_mask.tolist(), flush=True)

        # print('win calculated logp', (batch['ref_win_per_token_logp'] * win_weighted_mask).sum(-1), flush=True)
        # print('rej calculated logp', (batch['ref_rej_per_token_logp'] * rej_weighted_mask).sum(-1), flush=True)

        # print('win ref_win_per_token_logp', batch['ref_win_per_token_logp'], flush=True)
        # print('rej ref_rej_per_token_logp', batch['ref_rej_per_token_logp'], flush=True)

        # print('win logp', batch['ref_win_logp'], flush=True)
        # print('rej logp', batch['ref_rej_logp'], flush=True)
        # print('win avg logp', batch['ref_win_avg_logp'], flush=True)
        # print('rej avg logp', batch['ref_rej_avg_logp'], flush=True)
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          multimodal_cfg=dict(
                                              is_multimodal=data_args.is_multimodal,
                                              image_token_len=data_args.image_token_len,
                                              image_folder=data_args.image_folder,
                                              image_aspect_ratio=data_args.image_aspect_ratio,
                                              use_im_start_end=getattr(
                                                  data_args, 'mm_use_im_start_end', False),
                                              image_processor=getattr(
                                                  data_args, 'train_image_processor', None),
                                              data_source_names=getattr(
                                                  data_args, 'data_source_names'),
                                              data_source_weights=getattr(data_args, 'data_source_weights')))
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_dpo_data_module(tokenizer, data_args):
    train_dataset = DPODataset(tokenizer=tokenizer,
                               multimodal_cfg=dict(
                                   is_multimodal=data_args.is_multimodal,
                                   image_token_len=data_args.image_token_len,
                                   image_folder=data_args.image_folder,
                                   image_aspect_ratio=data_args.image_aspect_ratio,
                                   use_im_start_end=getattr(
                                       data_args, 'mm_use_im_start_end', False),
                                   image_processor=getattr(
                                       data_args, 'train_image_processor', None),
                                   data_source_names=getattr(
                                       data_args, 'data_source_names'),
                                   data_source_weights=getattr(data_args, 'data_source_weights')))
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForDPODataset(
        tokenizer=tokenizer, beta=data_args.dpo_beta, mod_token_weight=data_args.dpo_token_weight)

    if data_args.eval_data_source_names is not None:
        eval_datasets = {}
        for name in data_args.eval_data_source_names:
            eval_dataset = DPODataset(tokenizer=tokenizer,
                                      multimodal_cfg=dict(
                                          is_multimodal=data_args.is_multimodal,
                                          image_token_len=data_args.image_token_len,
                                          image_folder=data_args.image_folder,
                                          image_aspect_ratio=data_args.image_aspect_ratio,
                                          use_im_start_end=getattr(
                                              data_args, 'mm_use_im_start_end', False),
                                          image_processor=getattr(
                                              data_args, 'test_image_processor', None),
                                          data_source_names=[name],
                                          data_source_weights=[1]))
            eval_datasets[name] = eval_dataset
    else:
        eval_datasets = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_datasets,
                data_collator=data_collator)


def init_model(model_args, data_args, training_args):
    if model_args.vision_tower is not None:
        model = Beit3LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    model.config.use_cache = False

    if (hasattr(model.config, "mm_vision_tower") and
        model_args.vision_tower is not None and
            model_args.vision_tower != model.config.mm_vision_tower):

        print(
            f'Update vision arch from {model.config.mm_vision_tower} to {model_args.vision_tower}', flush=True)
        model.config.mm_vision_tower = model_args.vision_tower

        # may interpolate
        state_dict = interpolate_beit3(
            model.model.vision_tower, model_args.vision_tower)
        new_vision_tower = timm.create_model(model_args.vision_tower)
        new_vision_tower.load_state_dict(state_dict)
        model.model.vision_tower = new_vision_tower

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.vision_tower is not None:
        model_vision_dict = model.model.initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            no_randaug=training_args.no_randaug,
            num_query=model_args.num_query,
        )
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        if training_args.fully_tune:
            dtype = torch.float32
        model.model.vision_tower.to(dtype=dtype, device=training_args.device)
        vision_config = model_vision_dict['vision_config']

        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.train_image_processor, data_args.test_image_processor = model_vision_dict[
            'image_processor']
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.fully_tune = training_args.fully_tune
        assert model_args.tune_mm_mlp_adapter ^ model.config.fully_tune, f'Value of fully_tune and tune_mm_mlp_adapter are same: {model_args.tune_mm_mlp_adapter} {model.config.fully_tune}'
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.model.mm_projector.parameters():
                p.requires_grad = True

            model.model.query.requires_grad_(True)
            beit3 = model.model.vision_tower.beit3
            beit3.requires_grad_(True)
            if training_args.deepspeed:
                beit3.vision_embed.requires_grad_(False)
                beit3.apply(stop_gradient_by_name('A'))
            else:
                # with torch DDP, set zero LR rather than stop gradient
                # to accelerate training
                pass
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end

        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter)
        if training_args.task == 'OCR':
            special_token_list = ['<{}>'.format(idx) for idx in range(1024)]
            tokenizer.add_tokens(
                ['<quad>', '</quad>', '<ref>', '</ref>'], special_tokens=True)
            tokenizer.add_tokens(special_token_list, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))
        if model.config.fully_tune:
            model.requires_grad_(True)

        # remove unused params
        model.model.vision_tower.beit3.vision_embed.mask_token = None
        model.model.vision_tower.beit3.text_embed = None

        params_no_grad = [
            n for n, p in model.named_parameters() if not p.requires_grad]
        if is_main_process():
            print(f'No grad params are : {params_no_grad}', flush=True)
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(
                        len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(
                        len(params_no_grad), ', '.join(params_no_grad[:10])))
                print(
                    "[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    if training_args.task == 'LM' or training_args.task == 'OCR':
        data_module = make_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args)
    elif training_args.task == 'DPO':
        data_module = make_dpo_data_module(tokenizer, data_args=data_args)
    return model.cuda(), data_module, tokenizer


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.report_to == 'wandb':
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(['.cache', '_temp'])

    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [
        int(x) for x in data_args.data_source_weights.split('#')]

    data_args.eval_data_source_names = data_args.eval_data_source_names.split(
        '#')

    training_args.output_dir = training_args.output_dir.replace(
        'dpo_preference', 'dpo').replace('no_crop_vqa', 'ncrp_vqa')

    model, data_module, tokenizer = init_model(
        model_args, data_args, training_args)

    if training_args.task == 'LM' or training_args.task == 'OCR':
        trainer = MuffinTrainer(model=model,
                                tokenizer=tokenizer,
                                args=training_args,
                                **data_module)
    elif training_args.task == 'DPO':
        trainer = MuffinDPOTrainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   **data_module)

    # print(f'Training args: {training_args}')
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f'Resume from checkpoint.')
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f'Train from start.')
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
