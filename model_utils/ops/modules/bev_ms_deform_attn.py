# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction, ms_deform_pool_core

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


class BEVMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, q_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param q_model      query number of feature
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 128

        self.d_model = d_model
        self.q_model = q_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # self.attention_weights_h_0 = nn.Linear(d_model, 11)
        # self.attention_weights_h_1 = nn.Linear(d_model, 5)
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.ModuleList([nn.Linear(128, 128), nn.Linear(128, 128), nn.Linear(64, 128)])



        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # constant_(self.attention_weights_h_0.weight.data, 0.0)
        # constant_(self.attention_weights_h_0.bias.data, 0.0)
        # constant_(self.attention_weights_h_1.weight.data, 0.0)
        # constant_(self.attention_weights_h_1.bias.data, 0.0)
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.0)
        for net in self.key_proj:
            xavier_uniform_(net.weight.data)
            constant_(net.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        ms_lidar_features, #[bev, x_conv4.dense(), x_conv3.dense()]
        input_spatial_shapes,#[[200, 176], [200, 176], [400, 352]]
        input_padding_mask=None,

    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        # N, Len_in, _ = input_flatten.shape
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        # assert input_spatial_shapes.shape[0] == self.n_levels

        # value = self.value_proj(input_flatten)
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        # value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)


        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        # attention_weights = self.attention_weights(weight_query).view(
        #     N, Len_q, self.n_heads, self.n_levels * self.n_points
        # )

        # attention_weights_hlevel_1 = self.attention_weights_h_1(query)
        # attention_weights_hlevel_1 = F.softmax(attention_weights_hlevel_1, -1) #5
        # attention_weights_hlevel_0 = self.attention_weights_h_0(query)
        # attention_weights_hlevel_0 = F.softmax(attention_weights_hlevel_0, -1) #11
        # attn_weights_h_list = [attention_weights_hlevel_1, attention_weights_hlevel_0]
        attn_weights_h_list = []

        # attention_weights = F.softmax(attention_weights, -1).view(
        #     N, Len_q, self.n_heads, self.n_levels, self.n_points
        # )
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        output = ms_deform_pool_core(query, self.key_proj, sampling_locations, attn_weights_h_list,
                                                     ms_lidar_features)






        output = self.output_proj(output)
#query   B,200,128
#value_spatial_shapes  torch[[200,176],[200,176],[400,352]]
#sampling_locations B,200,8,3,4,2  x,y
#attention_weights_hlevel_1  B,200,5
#attention_weights_hlevel_0  B,200,11
#attn_weights_h_list [attention_weights_hlevel_1, attention_weights_hlevel_0]
#ms_lidar_features
        return output.permute(1, 0, 2).contiguous()
