#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM, \
    CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import timm
import muffin.model.beit3
from timm.models.layers import trunc_normal_
from muffin.model.utils import build_transform


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def construct_query_parameter(query_k, h_size, init_weights, init=True):
    query_data = torch.zeros(query_k, h_size)
    if init:
        trunc_normal_(query_data, std=.02)
    for idx in range(query_k):
        if init_weights[idx] is not None:
            query_data[idx] = init_weights[idx]
    query = torch.nn.Parameter(query_data)
    return query


# The implementation code is modified from DeiT (https://github.com/facebookresearch/deit.git)
def load_model_and_may_interpolate(checkpoint_model, model):
    state_dict = model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
        if pos_embed_key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_embed_key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                # being consistent with Fairseq, which starts from 2 for position embedding
                torchscale_model = True
                num_patches = model.beit3.vision_embed.num_patches
                num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - \
                    num_patches
            else:
                torchscale_model = False
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = getattr(
                    model, pos_embed_key).shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int(
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
                if torchscale_model:
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(
                        0)
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                else:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size,
                                                orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                if torchscale_model:
                    new_pos_embed = new_pos_embed.squeeze(0)
                checkpoint_model[pos_embed_key] = new_pos_embed
    return checkpoint_model


class Beit3ContrastiveLlavaConfig(LlamaConfig):
    model_type = "beit3_contrastive_llava"


class Beit3ContrastiveLlavaLlamaModel(LlamaModel):
    config_class = Beit3ContrastiveLlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(Beit3ContrastiveLlavaLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # overload mm_vision_tower if value is passed
            if mm_vision_tower is None:
                mm_vision_tower = config.mm_vision_tower
            vision_tower = timm.create_model(mm_vision_tower)
            state_dict = torch.load(
                '/mnt/data/user/tc_agi/multi_modal/checkpoints/beit-v3/beit3_large_itc_patch16_224.pth', map_location='cuda')['model']
            state_dict = load_model_and_may_interpolate(
                state_dict, vision_tower)
            vision_tower.load_state_dict(state_dict, strict=False)
            self.vision_tower = [vision_tower]

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(
                self.vision_tower[0].hidden_size, config.hidden_size)

        self.vision_config = lambda x: None

    def initialize_vision_modules(self, vision_tower, no_randaug, img_feat_layer):
        self.config.mm_vision_tower = vision_tower
        self.config.use_mm_proj = True
        self.config.img_feat_layer = img_feat_layer
        self.config.num_image_patch = 196

        if not hasattr(self, 'vision_tower'):
            vision_tower = timm.create_model(vision_tower)
            state_dict = torch.load(
                '/mnt/data/user/tc_agi/multi_modal/checkpoints/beit-v3/beit3_large_itc_patch16_224.pth', map_location='cuda')['model']
            state_dict = load_model_and_may_interpolate(
                state_dict, vision_tower)
            vision_tower.load_state_dict(state_dict, strict=False)
        else:
            vision_tower = self.vision_tower[0]
        self.vision_tower = [vision_tower.to(torch.float16)]

        train_img_transform, eval_img_transform = build_transform(
            is_train=True, randaug=not no_randaug, input_size=vision_tower.args.img_size), build_transform(is_train=False, input_size=vision_tower.args.img_size)

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(
                vision_tower.hidden_size, self.config.hidden_size)

        return dict(
            image_processor=(train_img_transform, eval_img_transform),
            image_token_len=self.config.num_image_patch,
            vision_config=self.vision_config
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:

            if isinstance(vision_tower, (list, )):
                vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(pixel_values=image.unsqueeze(
                            0), encode_image=True, img_feat_layer=self.config.img_feat_layer)
                        image_features.append(image_forward_out)
                else:
                    image_features = vision_tower(
                        pixel_values=images, encode_image=True, img_feat_layer=self.config.img_feat_layer)
            if type(images) is list:
                image_features = [self.mm_projector(
                    image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(
                self.config.num_image_patch, vision_tower.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + \
                        (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if self.vision_config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(
                        cur_input_ids == self.vision_config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                            raise ValueError(
                                "The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features,
                                                             cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat(
                                (cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == self.vision_config.im_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(
                        cur_input_ids == self.vision_config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError(
                            "The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(
                        ), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(Beit3ContrastiveLlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class Beit3ContrastiveLlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = Beit3ContrastiveLlavaConfig

    def __init__(self, config, mm_vision_tower=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = Beit3ContrastiveLlavaLlamaModel(
            config, mm_vision_tower=mm_vision_tower)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False):
        self.model.vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.model.vision_config.im_start_token, self.model.vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        self.model.vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]


AutoConfig.register("beit3_contrastive_llava", Beit3ContrastiveLlavaConfig)
AutoModelForCausalLM.register(
    Beit3ContrastiveLlavaConfig, Beit3ContrastiveLlavaLlamaForCausalLM)
