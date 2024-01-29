from torch import nn
from torch.utils.data.sampler import Sampler, RandomSampler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import os
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import logger
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, has_length
from transformers.trainer_pt_utils import nested_numpify, find_batch_size
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from utils.utils import is_main_process

from muffin.eval.muffin_inference_logp import get_batch_logps


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class MuffinTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [
            name for name in decay_parameters if "bias" not in name]

        def should_zero_lr(param_name: str):
            if 'beit3' in param_name:
                if '.A' in param_name:
                    return True
                if 'beit3.vision_embed' in param_name:
                    return True
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (p.requires_grad and should_zero_lr(n))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": 0.0,
                "initial_lr": 0.0
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad and not should_zero_lr(n))
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad and not should_zero_lr(n))
                ],
                "weight_decay": 0.0,
            },
        ]
        for n, p in model.named_parameters():
            # print(f'Check LR of {n}')
            if should_zero_lr(n) and is_main_process():
                print(f'Zero LR params: {n}', flush=True)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args)
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs)

        self.scheduler = self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer)
        print(f'LR schduler is ', self.scheduler)


class ChunckedRandomSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size=5000) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        print(f'Chuncked Random Sampler seed is {seed}')
        generator = torch.Generator()
        generator.manual_seed(seed)

        for st in torch.randperm(n // self.chunk_size, generator=generator).tolist():
            base = st * self.chunk_size
            for i in torch.randperm(self.chunk_size, generator=generator).tolist():
                yield base + i

        base = (n // self.chunk_size) * self.chunk_size
        for i in torch.randperm(n % self.chunk_size, generator=generator).tolist():
            yield base + i

    def __len__(self) -> int:
        return len(self.data_source)


class ZephyrTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        # Build the sampler.
        if self.args.group_by_length:
            assert NotImplementedError
        else:
            if len(self.train_dataset) >= 10_000_000:
                return ChunckedRandomSampler(self.train_dataset)
            else:
                return RandomSampler(self.train_dataset)


def kto_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.
    For each batch of n/2 chosen examples and n/2 rejected examples (belonging to n different inputs), calculate the loss as follows.

    If generation y ~ p_chosen, where x' ~ are the examples with rejected generations, we have the 'chosen' loss:
        L(x, y) := 1 - sigmoid(beta * (log p_policy(y|x) - log p_reference(y|x) - KL(p_policy(y_rejected|x') || p_reference(y_rejected|x')))
    If generation y ~ p_rejected, , where x' ~ are the examples with chosen generations, we have the 'rejected' loss:
        L(x, y) := 1 - sigmoid(beta * KL(p_policy(y_chosen|x') || p_reference(y_chosen|x')) - [log p_policy(y|x) - log p_reference(y|x)])
    """
    chosen_KL = (policy_chosen_logps -
                 reference_chosen_logps).mean().clamp(min=0)
    rejected_KL = (policy_rejected_logps -
                   reference_rejected_logps).mean().clamp(min=0)

    chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
    rejected_logratios = (policy_rejected_logps - reference_rejected_logps)

    losses = torch.cat((1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                        1 - F.sigmoid(beta * (chosen_KL - rejected_logratios))), 0)

    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )

    if token_weighted:
        token_log_prob = get_batch_logps(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = get_batch_logps(
            output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob


def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


def get_beta_and_logps(data_dict, model, args):
    win_input_ids = data_dict.pop('win_input_ids')
    rej_input_ids = data_dict.pop('rej_input_ids')

    win_labels = data_dict.pop('win_labels')
    rej_labels = data_dict.pop('rej_labels')

    # win_attention_mask = data_dict.pop('win_attention_mask')
    # rej_attention_mask = data_dict.pop('rej_attention_mask')

    ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
    ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
    ref_win_logp = data_dict.pop('ref_win_logp')
    ref_rej_logp = data_dict.pop('ref_rej_logp')
    ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
    ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
    if args.dpo_use_average:
        ref_win_logp = ref_win_avg_logp
        ref_rej_logp = ref_rej_avg_logp

    beta = data_dict.pop('beta')
    images = data_dict.pop('images')

    concatenated_input_ids = data_dict.pop('concatenated_input_ids')
    concatenated_labels = data_dict.pop('concatenated_labels')
    # concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
    concatenated_attention_mask = None
    concatenated_images = torch.cat([images, images], dim=0)

    win_token_weight = data_dict.pop('win_token_weight')
    rej_token_weight = data_dict.pop('rej_token_weight')
    concatenated_token_weight = data_dict.pop('concatenated_token_weight')

    concatenated_logp = forward_DPO(model,
                                    concatenated_input_ids,
                                    concatenated_labels,
                                    concatenated_attention_mask,
                                    concatenated_images,
                                    token_weighted=args.dpo_token_weighted,
                                    dpo_use_average=args.dpo_use_average,
                                    **data_dict)
    win_size = win_input_ids.shape[0]
    rej_size = rej_input_ids.shape[0]
    assert win_size == rej_size

    if args.dpo_token_weighted:
        # print(f'compute_loss win {win_input_ids.shape} {win_labels.shape} {ref_win_per_token_logp.shape} {win_token_weight.shape}', flush=True)
        # print(f'compute_loss rej {rej_input_ids.shape} {rej_labels.shape} {ref_rej_per_token_logp.shape} {rej_token_weight.shape}', flush=True)
        # print(f'compute_loss cat {concatenated_input_ids.shape} {concatenated_labels.shape} {concatenated_logp.shape} {concatenated_token_weight.shape}', flush=True)

        # for i in range(len(ref_win_per_token_logp)):
        #     print(f'compuate loss {i} win_input_ids={win_input_ids[i]}\nwin_labels={win_labels[i]}\nwin_per_token_logp={ref_win_per_token_logp[i]}\nwin_token_weight={win_token_weight[i]}\n', flush=True)
        #     print(f'compuate loss {i} rej_input_ids={rej_input_ids[i]}\nrej_labels={rej_labels[i]}\nrej_per_token_logp={ref_rej_per_token_logp[i]}\nrej_token_weight={rej_token_weight[i]}\n', flush=True)
        ref_win_logp = compute_weighted_logp(
            ref_win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
        ref_rej_logp = compute_weighted_logp(
            ref_rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
        concatenated_logp = compute_weighted_logp(
            concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)

        if torch.any(torch.isnan(ref_win_logp)):
            print(f'ref_win_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(ref_rej_logp)):
            print(f'ref_rej_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(concatenated_logp)):
            print(f'concatenated_logp fail', flush=True)
            exit()

    policy_win_logp, policy_rej_logp = concatenated_logp.split(
        [win_size, rej_size])
    return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta


class ZephyrDPOTrainer(ZephyrTrainer):
    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args)

        if self.args.past_index >= 0:
            raise NotImplementedError

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(
            chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(
            rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(
            reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - \
            metrics[f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(
            policy_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(
            policy_win_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_rejected'] = self._nested_gather(
            ref_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_chosen'] = self._nested_gather(
            ref_win_logp.mean()).mean().item()
        self.log(metrics)

        return loss
    def compute_eval(self, model: Module, inputs: dict):
        inputs = self._prepare_inputs(inputs)

        data_dict = inputs

        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args)

        if self.args.past_index >= 0:
            raise NotImplementedError

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        cls_accuracies = (policy_win_logp > policy_rej_logp).float()

        return chosen_rewards, rejected_rewards, reward_accuracies, cls_accuracies, losses

    def custom_compute_metrics(self, chosen_rewards, rejected_rewards, reward_accuracies, cls_accuracies, losses):
        metrics = {}

        reward_accuracies = float(reward_accuracies.mean())
        cls_accuracies = float(cls_accuracies.mean())
        losses = float(losses.mean())
        
        chosen = float(chosen_rewards.mean())
        rejected = float(rejected_rewards.mean())
        margin = chosen - rejected

        metrics['eval_cls_acc'] = cls_accuracies
        metrics['eval_reward_acc'] = reward_accuracies
        metrics['eval_reward_margin'] = margin
        metrics['eval_loss'] = losses

        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        model.eval()

        with torch.no_grad():
            self.callback_handler.eval_dataloader = dataloader
            # Do this before wrapping.
            eval_dataset = getattr(dataloader, "dataset", None)

            if args.past_index >= 0:
                self._past = None

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            chosen_rewards_host = None
            rejected_rewards_host = None
            reward_accuracies_host = None
            cls_accuracies_host = None
            losses_host = None

            # losses/preds/labels on CPU (final containers)
            all_chosen_rewards = None
            all_rejected_rewards = None
            all_reward_accuracies = None
            all_cls_accuracies = None
            all_losses = None
            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                chosen_reward, rejected_reward, reward_accuracy, cls_accuracy, loss = self.compute_eval(
                    model, inputs)

                # Update containers on host
                if chosen_reward is not None:
                    chosen_rewards = self._nested_gather(chosen_reward)
                    chosen_rewards_host = chosen_rewards if chosen_rewards_host is None else torch.cat(
                        (chosen_rewards_host, chosen_rewards), dim=0)
                if rejected_reward is not None:
                    rejected_rewards = self._nested_gather(rejected_reward)
                    rejected_rewards_host = rejected_rewards if rejected_rewards_host is None else torch.cat(
                        (rejected_rewards_host, rejected_rewards), dim=0)
                if reward_accuracy is not None:
                    reward_accuracies = self._nested_gather(reward_accuracy)
                    reward_accuracies_host = reward_accuracies if reward_accuracies_host is None else torch.cat(
                        (reward_accuracies_host, reward_accuracies), dim=0)
                if cls_accuracy is not None:
                    cls_accuracies = self._nested_gather(cls_accuracy)
                    cls_accuracies_host = cls_accuracies if cls_accuracies_host is None else torch.cat((cls_accuracies_host, cls_accuracies), dim=0)
                if loss is not None:
                    losses = self._nested_gather(loss)
                    losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

                self.control = self.callback_handler.on_prediction_step(
                    args, self.state, self.control)

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

            # Gather all remaining tensors and put them back on the CPU
            if chosen_rewards_host is not None:
                chosen_rewards = nested_numpify(chosen_rewards_host)
                all_chosen_rewards = chosen_rewards if all_chosen_rewards is None else np.concatenate(
                    (all_chosen_rewards, chosen_rewards), axis=0)
            if rejected_rewards_host is not None:
                rejected_rewards = nested_numpify(rejected_rewards_host)
                all_rejected_rewards = rejected_rewards if all_rejected_rewards is None else np.concatenate(
                    (all_rejected_rewards, rejected_rewards), axis=0)
            if reward_accuracies_host is not None:
                reward_accuracies = nested_numpify(reward_accuracies_host)
                all_reward_accuracies = reward_accuracies if all_reward_accuracies is None else np.concatenate(
                    (all_reward_accuracies, reward_accuracies), axis=0)
            if cls_accuracies_host is not None:
                cls_accuracies = nested_numpify(cls_accuracies_host)
                all_cls_accuracies = cls_accuracies if all_cls_accuracies is None else np.concatenate((all_cls_accuracies, cls_accuracies), axis=0)
            if losses_host is not None:
                losses = nested_numpify(losses_host)
                all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            
            # Number of samples
            num_samples = len(eval_dataset)

            # Metrics!
            metrics = self.custom_compute_metrics(
                all_chosen_rewards, all_rejected_rewards, all_reward_accuracies, all_cls_accuracies, all_losses)

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            if hasattr(self, "jit_compilation_time"):
                metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}/"):
                    metrics[f"{metric_key_prefix}/{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


class MuffinDPOTrainer(MuffinTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args)
        # print(f'policy logp: {policy_win_logp}, {policy_rej_logp}')

        if self.args.past_index >= 0:
            raise NotImplementedError

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # loss = losses.mean()

        # do SFT
        # loss = - policy_win_logp.mean()
        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()
        # loss = DPO_weight * losses.mean() - SFT_weight * policy_rej_logp.mean()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(
            chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(
            rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(
            reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - \
            metrics[f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(
            policy_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(
            policy_win_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_rejected'] = self._nested_gather(
            ref_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_chosen'] = self._nested_gather(
            ref_win_logp.mean()).mean().item()
        # metrics[f'batch_size'] = len(win_labels)
        self.log(metrics)

        return loss

    def compute_eval(self, model: Module, inputs: dict):
        inputs = self._prepare_inputs(inputs)

        data_dict = inputs

        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args)

        if self.args.past_index >= 0:
            raise NotImplementedError

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        return chosen_rewards, rejected_rewards, reward_accuracies

    def custom_compute_metrics(self, chosen_rewards, rejected_rewards, reward_accuracies):
        metrics = {}

        reward_accuracies = float(reward_accuracies.mean())
        chosen = float(chosen_rewards.mean())
        rejected = float(rejected_rewards.mean())
        margin = chosen - rejected

        metrics['eval_reward_acc'] = reward_accuracies
        metrics['eval_reward_margin'] = margin

        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        # logger?
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        with torch.no_grad():
            self.callback_handler.eval_dataloader = dataloader
            # Do this before wrapping.
            eval_dataset = getattr(dataloader, "dataset", None)

            if args.past_index >= 0:
                self._past = None

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            chosen_rewards_host = None
            rejected_rewards_host = None
            reward_accuracies_host = None

            # losses/preds/labels on CPU (final containers)
            all_chosen_rewards = None
            all_rejected_rewards = None
            all_reward_accuracies = None
            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                chosen_reward, rejected_reward, reward_accuracy = self.compute_eval(
                    model, inputs)

                # Update containers on host
                if chosen_reward is not None:
                    chosen_rewards = self._nested_gather(chosen_reward)
                    chosen_rewards_host = chosen_rewards if chosen_rewards_host is None else torch.cat(
                        (chosen_rewards_host, chosen_rewards), dim=0)
                if rejected_reward is not None:
                    rejected_rewards = self._nested_gather(rejected_reward)
                    rejected_rewards_host = rejected_rewards if rejected_rewards_host is None else torch.cat(
                        (rejected_rewards_host, rejected_rewards), dim=0)
                if reward_accuracy is not None:
                    reward_accuracies = self._nested_gather(reward_accuracy)
                    reward_accuracies_host = reward_accuracies if reward_accuracies_host is None else torch.cat(
                        (reward_accuracies_host, reward_accuracies), dim=0)

                self.control = self.callback_handler.on_prediction_step(
                    args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                    if chosen_rewards_host is not None:
                        chosen_rewards = nested_numpify(chosen_rewards_host)
                        all_chosen_rewards = chosen_rewards if all_chosen_rewards is None else np.concatenate(
                            (all_chosen_rewards, chosen_rewards), axis=0)
                    if rejected_rewards_host is not None:
                        rejected_rewards = nested_numpify(
                            rejected_rewards_host)
                        all_rejected_rewards = rejected_rewards if all_rejected_rewards is None else np.concatenate(
                            (all_rejected_rewards, rejected_rewards), axis=0)
                    if reward_accuracies_host is not None:
                        reward_accuracies = nested_numpify(
                            reward_accuracies_host)
                        all_reward_accuracies = reward_accuracies if all_reward_accuracies is None else np.concatenate(
                            (all_reward_accuracies, reward_accuracies), axis=0)

                    # Set back to None to begin a new accumulation
                    chosen_rewards_host, rejected_rewards_host, reward_accuracies_host = None, None, None

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

            # Gather all remaining tensors and put them back on the CPU
            if chosen_rewards_host is not None:
                chosen_rewards = nested_numpify(chosen_rewards_host)
                all_chosen_rewards = chosen_rewards if all_chosen_rewards is None else np.concatenate(
                    (all_chosen_rewards, chosen_rewards), axis=0)
            if rejected_rewards_host is not None:
                rejected_rewards = nested_numpify(rejected_rewards_host)
                all_rejected_rewards = rejected_rewards if all_rejected_rewards is None else np.concatenate(
                    (all_rejected_rewards, rejected_rewards), axis=0)
            if reward_accuracies_host is not None:
                reward_accuracies = nested_numpify(reward_accuracies_host)
                all_reward_accuracies = reward_accuracies if all_reward_accuracies is None else np.concatenate(
                    (all_reward_accuracies, reward_accuracies), axis=0)

            # Number of samples
            num_samples = len(eval_dataset)

            # Metrics!
            metrics = self.custom_compute_metrics(
                all_chosen_rewards, all_rejected_rewards, all_reward_accuracies)

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            if hasattr(self, "jit_compilation_time"):
                metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)
