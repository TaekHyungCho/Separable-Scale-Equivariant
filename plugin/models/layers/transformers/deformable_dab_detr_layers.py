# Copyright (c) OpenMMLab. All rights reserved.
from typing import List,Optional,Tuple

import torch
import torch.nn as nn
import math
from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.models.layers import (DeformableDetrTransformerDecoder,DeformableDetrTransformerDecoderLayer)
from mmdet.models.layers.transformer.utils import (MLP,inverse_sigmoid)


class Deformable_DABDetrTransformerDecoder(DeformableDetrTransformerDecoder):
    """Decoder of DAB-DETR.

    Args:
        query_dim (int): The last dimension of query pos,
            4 for anchor format, 2 for point format.
            Defaults to 4.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        with_modulated_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
    """

    def __init__(self,
                 *args,
                 query_dim: int = 4,
                 query_scale_type: str = 'cond_elewise',
                 with_modulated_hw_attn: bool = False,
                 **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.with_modulated_hw_attn = with_modulated_hw_attn

        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize decoder layers and other layers."""
        assert self.query_dim in [2, 4], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims
        #self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP(self.query_dim // 2 * embed_dims, embed_dims,
                                  embed_dims, 2)
        #self.ref_point_head = MLP(self.query_dim, embed_dims, embed_dims, 3)

        if self.with_modulated_hw_attn and self.query_dim == 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        # self.keep_query_pos = self.layers[0].keep_query_pos
        # if not self.keep_query_pos:
        #     for layer_id in range(self.num_layers - 1):
        #         self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim).
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.

        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4).
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            
            query_sine_embed = self.gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
            raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256

            #raw_query_pos = self.ref_point_head(reference_points_input) # bs, nq, 256
            
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            query_pos = pos_transformation * raw_query_pos

            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
       
    
    def gen_sineembed_for_position(self,pos_tensor):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos


# class Deformable_DABDetrTransformerEncoder(DeformableDetrTransformerEncoder):
#     """Encoder of DAB-DETR."""

#     def _init_layers(self):
#         """Initialize encoder layers."""
#         self.layers = ModuleList([
#             DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         embed_dims = self.layers[0].embed_dims
#         self.embed_dims = embed_dims
#         self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)

#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, spatial_shapes: Tensor,
#                 level_start_index: Tensor, valid_ratios: Tensor,
#                 **kwargs) -> Tensor:
#         """Forward function of encoder.

#         Args:
#             query (Tensor): Input queries of encoder, has shape
#                 (bs, num_queries, dim).
#             query_pos (Tensor): The positional embeddings of the queries, has
#                 shape (bs, num_feat_points, dim).
#             key_padding_mask (Tensor): ByteTensor, the key padding mask
#                 of the queries, has shape (bs, num_feat_points).

#         Returns:
#             Tensor: With shape (num_queries, bs, dim).
#         """
#         reference_points = self.get_encoder_reference_points(
#             spatial_shapes, valid_ratios, device=query.device)

#         for layer in self.layers:
#             pos_scales = self.query_scale(query)
#             query = layer(
#                 query,
#                 query_pos=query_pos * pos_scales,
#                 key_padding_mask=key_padding_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 valid_ratios=valid_ratios,
#                 reference_points=reference_points,
#                 **kwargs)

#         return query
