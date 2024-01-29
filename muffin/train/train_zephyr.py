import os
import gc
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
from muffin.train.trainers import ZephyrTrainer, ZephyrDPOTrainer
from muffin.data.datasets import SingleDataSourceDataset, MultiDataSourceDataset
from muffin.data.data_processors import register_data_path
from muffin.train.train_utils import SFT_collator_fn, IGNORE_INDEX, encode_multimodal_sample, encode_multimodal_preference_sample, zephyr_encode_multimodal_sample, zephyr_preprocess

from muffin.model.zephyr_mm import ZephyrMMForCausalLM
from muffin.train.train_muffin import DataCollatorForDPODataset

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
    num_query: int = 256


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
    tune_CLIP: bool = field(default=False)
    image_size: int = 448


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

        data_dict = zephyr_encode_multimodal_sample(
            source, self.tokenizer, self.multimodal_cfg)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return SFT_collator_fn(instances, self.tokenizer.pad_token_id)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          multimodal_cfg=dict(
                                              is_multimodal=data_args.is_multimodal,
                                              image_token_len=data_args.image_token_len,
                                              image_folder=data_args.image_folder,
                                              image_aspect_ratio=data_args.image_aspect_ratio,
                                              use_im_start_end=True,
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
            source, self.tokenizer, self.multimodal_cfg, preprocess_func=zephyr_preprocess)
        return rej_data_dict, win_data_dict


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
        model = ZephyrMMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=True,
            tune_clip=training_args.tune_CLIP
        )
    if training_args.tune_CLIP and not hasattr(model.model.config, 'tune_CLIP'):
        print(f'Tune CLIP from eva weight')
        state_dict = torch.load(
            '/data/public/multimodal/multimodal_model_ckpts/timm/eva02_enormous_patch14_clip_224.laion2b_plus.pt')
        model.model.vision_tower.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        pass
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        truncation_side='right',
    )

    if model_args.vision_tower is not None:
        model_vision_dict = model.model.initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            no_randaug=training_args.no_randaug,
            num_query=model_args.num_query,
            image_size=training_args.image_size,
            tune_clip=training_args.tune_CLIP,
        )
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        # if training_args.fully_tune:
        #     dtype = torch.float32

        vision_tower_module = model.model.vision_tower[0] if isinstance(
            model.model.vision_tower, list) else model.model.vision_tower
        vision_tower_module.to(dtype=dtype, device=training_args.device)
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
            for p in model.model.resampler.parameters():
                p.requires_grad = True

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter)
        if model.config.fully_tune:
            model.requires_grad_(True)

        params_no_grad = [
            n for n, p in model.named_parameters() if not p.requires_grad]
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

    if training_args.tune_CLIP:
        print(f'=======> Tune CLIP <=========')
        model.model.vision_tower = vision_tower_module
        model.model.vision_tower.requires_grad_(True)
        model.model.vision_tower.to(
            dtype=torch.bfloat16, device=training_args.device)
        model.to(dtype=torch.bfloat16, device=training_args.device)
        model.model.config.tune_CLIP = True
    params_no_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad]
    if is_main_process():
        print(f'No grad params are : {params_no_grad}', flush=True)

    if training_args.task == 'LM':
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

    assert model_args.mm_use_im_start_end == True

    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [
        int(x) for x in data_args.data_source_weights.split('#')]

    data_args.eval_data_source_names = data_args.eval_data_source_names.split(
        '#')

    model, data_module, tokenizer = init_model(
        model_args, data_args, training_args)

    if training_args.task == 'LM':
        trainer = ZephyrTrainer(model=model,
                                tokenizer=tokenizer,
                                args=training_args,
                                **data_module)
    elif training_args.task == 'DPO':
        trainer = ZephyrDPOTrainer(model=model,
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
