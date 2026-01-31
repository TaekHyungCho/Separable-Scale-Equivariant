# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

from mmengine.model import uniform_init
from torch import Tensor, nn
import torch

from mmdet.registry import MODELS
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer import inverse_sigmoid
from plugin.models.layers import Deformable_DABDetrTransformerDecoder
from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.models.layers.transformer import DeformableDetrTransformerEncoder


@MODELS.register_module()
class Deformable_DABDETR(DeformableDETR):
    r"""Implementation of `DAB-DETR:
    Dynamic Anchor Boxes are Better Queries for DETR.

    <https://arxiv.org/abs/2201.12329>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DAB-DETR>`_.

    Args:
        with_random_refpoints (bool): Whether to randomly initialize query
            embeddings and not update them during training.
            Defaults to False.
        num_patterns (int): Inspired by Anchor-DETR. Defaults to 0.
    """

    def __init__(self,
                 *args,
                 with_random_refpoints: bool = False,
                 num_patterns: int = 0,
                 **kwargs) -> None:
        self.with_random_refpoints = with_random_refpoints
        assert isinstance(num_patterns, int), \
            f'num_patterns should be int but {num_patterns}.'
        self.num_patterns = num_patterns

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = Deformable_DABDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_dim = self.decoder.query_dim
        self.tgt_embedding = nn.Embedding(self.num_queries,self.embed_dims)
        self.ref_embedding = nn.Embedding(self.num_queries,self.query_dim)
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(Deformable_DABDETR, self).init_weights(use_dab=True)

        if self.with_random_refpoints:
            uniform_init(self.ref_embedding)
            self.ref_embedding.weight.data[:, :2] = \
                inverse_sigmoid(self.ref_embedding.weight.data[:, :2])
            self.ref_embedding.weight.data[:, :2].requires_grad = False

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
       
        batch_size, _, c = memory.shape
        enc_outputs_class, enc_outputs_coord = None, None
        tgt_embed = self.tgt_embedding.weight
        ref_embed = self.ref_embedding.weight
        query_embed = torch.cat((tgt_embed,ref_embed),dim=1)
        query_pos = None
        query = query_embed[...,:self.embed_dims]
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = query_embed[...,self.embed_dims:].sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        
        assert query_pos is None

        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references)
        return decoder_outputs_dict