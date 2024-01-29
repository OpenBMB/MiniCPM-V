# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)

        # self.apply(self._init_weights) # no longer necessary since we only use the pre-trained ckpt
        # self.mim_head = nn.Linear(1024, 8192)
        self.num_img_patches = self.beit3.vision_embed.num_position_embeddings()
        self.hidden_size = args.encoder_embed_dim

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pixel_values, query_embed=None, encode_image=False, img_feat_layer=-1, attn_mask=None):
        assert (query_embed is not None) ^ encode_image
        B = pixel_values.size(0)
        dtype = self.beit3.vision_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype)
        token_embeddings = self.beit3.vision_embed(pixel_values)
        multiway_split_position = -1
        if query_embed is not None:
            query_embed = torch.stack([query_embed] * B)
            multiway_split_position = token_embeddings.size(1)
            token_embeddings = torch.cat(
                [token_embeddings, query_embed], dim=1)

        outputs = self.beit3.encoder(
            src_tokens=None,
            token_embeddings=token_embeddings,
            multiway_split_position=multiway_split_position,
            return_all_hiddens=encode_image,
            attn_mask=attn_mask,
        )
        vision_hidden_states = outputs["encoder_out"]
        if query_embed is not None:
            vision_hidden_states = vision_hidden_states[:,
                                                        self.num_img_patches:]
        if encode_image:
            vision_hidden_states = outputs['encoder_states'][img_feat_layer][:,
                                                                             1:self.num_img_patches]
        return vision_hidden_states


@register_model
def beit3_large_patch16_224(pretrained=False, **kwargs):
    args = _get_large_config(img_size=224, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_256(pretrained=False, **kwargs):
    args = _get_large_config(img_size=256, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_336(pretrained=False, **kwargs):
    args = _get_large_config(img_size=336, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_448(pretrained=False, **kwargs):
    args = _get_large_config(img_size=448, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_672(pretrained=False, **kwargs):
    args = _get_large_config(img_size=672, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model


@register_model
def beit3_large_itc_patch16_224(pretrained=False, **kwargs):
    args = _get_large_config(img_size=224, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model
