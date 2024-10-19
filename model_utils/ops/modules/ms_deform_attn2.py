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

from ..functions import MSDeformAttnFunction
from ..functions import ms_deform_attn_core_pytorch
from ...attentions import attn_dict

from scipy.spatial import cKDTree
from pynndescent import NNDescent
import concurrent.futures
from multiprocessing import Pool
from functools import partial
from pcdet.ops.knn.knn import knn

import time

# torch.multiprocessing.set_start_method('spawn')

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, q_model=256, n_levels=4, n_heads=8, n_points=4, n_heads_v=4, n_points_v=1,
                 q_method=None, q_rep_place=None, Dual_spatial=True):
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

        self.im2col_step = 64

        self.d_model = d_model
        self.q_model = q_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_heads_v = n_heads_v
        self.n_points_v = n_points_v

        self.project_Q = nn.Linear(d_model, d_model)
        self.project_K = nn.Linear(d_model, d_model)
        self.project_V = nn.Linear(d_model, d_model)

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets_v = nn.Linear(d_model, n_heads_v * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # self.attention_weights_v = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        # self.value_v_proj = nn.Linear(d_model, d_model)
        self.output_proj_i = nn.Linear(d_model, d_model)
        self.output_proj_v = nn.Linear(d_model, d_model)
        self.q_method = q_method
        self.q_rep_place = q_rep_place
        if q_method == 'gating':
            self.q_gating = attn_dict['BiGateSum1D_2'](d_model, d_model)
        self.use_pytorch_version = False

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


        # thetas_v = torch.arange(self.n_heads_v, dtype=torch.float32) * (
        #         2.0 * math.pi / self.n_heads_v
        # )
        # grid_init = torch.stack([thetas_v.cos(), thetas_v.sin()], -1)
        # grid_init = (
        #     (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        #     .view(self.n_heads_v, 1, 1, 3)
        #     .repeat(1, self.n_points_v, 1)
        # )
        # for i in range(self.n_points_v):
        #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offsets_v.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.sampling_offsets_v.weight.data)
        constant_(self.sampling_offsets_v.bias.data, 0.0)

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

        # constant_(self.attention_weights_v.weight.data, 0.0)
        # constant_(self.attention_weights_v.bias.data, 0.0)

        xavier_uniform_(self.project_Q.weight.data)
        constant_(self.project_Q.bias.data, 0.0)

        xavier_uniform_(self.project_K.weight.data)
        constant_(self.project_K.bias.data, 0.0)

        xavier_uniform_(self.project_V.weight.data)
        constant_(self.project_V.bias.data, 0.0)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)

        xavier_uniform_(self.output_proj_i.weight.data)
        constant_(self.output_proj_i.bias.data, 0.0)

        xavier_uniform_(self.output_proj_v.weight.data)
        constant_(self.output_proj_v.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        # reference_ld_points,#[B,N_v,1,3]
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        q_lidar_indices,
        input_padding_mask=None,
        i_query=None,
        input_flatten_v=None,
        Dual_spatial=True,
        set_center_xyz=None,
        group_features=None,
        q_lidar_grid=None
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
        N, Len_in, _ = input_flatten.shape

        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        assert input_spatial_shapes.shape[0] == self.n_levels

        value = self.value_proj(input_flatten)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        output_dict = {}
        # if Dual_spatial:
        #     # value_v = self.value_v_proj(input_flatten_v)
        #     # value_v = value_v.view(N, Len_q, self.n_heads, self.d_model // self.n_heads)
        #     Len_set, Len_p_perset = group_features.shape[2:]
        #     group_features_flip = group_features.permute(0, 2, 3, 1)
        #
        #     # [B,N_v,64]->[B,N_v,4,16]->[B,4,N_v,16]->[B*4,N_v,1,16]->[B*4*N_v,1,16]
        #     Q = self.project_Q(query).reshape(N, Len_q, self.n_heads_v, -1).\
        #         transpose(1, 2).flatten(0, 1).unsqueeze(2).flatten(0, 1)
        #     K = self.project_K(group_features_flip).reshape(N,Len_set,Len_p_perset,self.n_heads_v, -1).\
        #         permute(0, 3, 1, 2, 4).flatten(0, 1) # [B,2048,32,64]->[B,2048,32,4,16]->[B,4,2048,32,16]->[B*4,2048,32,16]
        #     V = self.project_V(group_features_flip).reshape(N,Len_set,Len_p_perset,self.n_heads_v, -1).\
        #         permute(0, 3, 1, 2, 4).flatten(0, 1) # [B,2048,32,64]


        weight_query = query.clone()
        if self.q_method is not None:
            assert i_query is not None
            assert self.q_rep_place is not None
            if self.q_method == 'gating':
                g_query, g_i_query = self.q_gating(query, i_query)
                # new_query = query * query_scale + i_query * i_query_scale
                new_query = g_query + g_i_query - query - i_query
            elif self.q_method == 'sum':
                new_query = query + i_query
            elif self.q_method == 'image':
                new_query = i_query
            else:
                raise NotImplementedError('q_method must be among ["gating", "sum", "image"]')

            if 'offset' in self.q_rep_place:
                query = new_query
            if 'weight' in self.q_rep_place:
                weight_query = new_query
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )#[B,N_v,8,1,4,2]

        attention_weights = self.attention_weights(weight_query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )#[B,N_v,8,1,4]

        # if Dual_spatial:
        #
        #     sampling_offsets_v = self.sampling_offsets_v(query).view(
        #         N, Len_q, self.n_heads_v, 3)  # [B,N_v,4,3]
        #     # offset_normalizer_v = torch.tensor([[70.4, 80, 4]]).cuda()
        #     sampling_xyz_v = q_lidar_grid[:, :, None, :] + sampling_offsets_v #[B,N_V,4,3]
        #
        #     set_idx = knn(1, set_center_xyz, sampling_xyz_v.permute(0,3,1,2).view(N, 3, -1), True)#[B,1,N_V*4]
        #     set_idx = set_idx.squeeze(1).view(N, -1, Len_q).flatten(0, 1)#[B*4,N_v]
        #     d_per_head = K.shape[-1]
        #     #[B*4*N_v,32,16]
        #     K_select = torch.gather(K, 1, set_idx.unsqueeze(-1).unsqueeze(-1).expand(N*self.n_heads_v, Len_q, Len_p_perset, d_per_head).long()).flatten(0,1)
        #     # [B*4*N_v,32,16]
        #     V_select = torch.gather(V, 1, set_idx.unsqueeze(-1).unsqueeze(-1).expand(N*self.n_heads_v, Len_q, Len_p_perset, d_per_head).long()).flatten(0,1)
        #
        #     attention_weights = torch.matmul(Q, K_select.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        #     attention_weights = F.softmax(attention_weights, -1)
        #     output_v = attention_weights.matmul(V_select) #[B*4*N_v,1,16]
        #     #[B*4*N_v,1,16]->[B,4,N_v,16]->[B,N_v,64]
        #     output_v = output_v.squeeze(1).reshape(N, self.n_heads_v, Len_q, d_per_head).permute(0, 2, 1, 3).flatten(-2, -1)
        #
        #     output_v = self.output_proj_v(output_v)  # [B,N_v,64]
        #     output_dict['output_v'] = output_v

            #
            # attention_weights_v = self.attention_weights_v(weight_query).view(
            #     N, Len_q, self.n_heads, self.n_levels * self.n_points
            # )
            # attention_weights_v = F.softmax(attention_weights_v, -1).view(
            #     N, Len_q, self.n_heads, self.n_levels, self.n_points
            # )#[B,N_v,8,1,4]



        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )#[B,N_v,8,1,4,2]

            # if Dual_spatial:
            #     offset_normalizer_v = torch.tensor([[176, 200, 4]]).cuda()
            #     sampling_locations_v = (
            #         reference_ld_points[:, :, None, :, :]
            #         + sampling_offsets_v / offset_normalizer_v[None, None, None, :, :]
            #     )#[B,N_v,8,2,3]

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



        if not self.use_pytorch_version:
            #更新的图像特征
            output_i = MSDeformAttnFunction.apply(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output_i = ms_deform_attn_core_pytorch(value,
                                                 input_spatial_shapes,
                                                 # offset_normalizer_v,#[176,200,4](xyz)
                                                 sampling_locations,
                                                 # sampling_locations_v,
                                                 # q_lidar_indices,
                                                 attention_weights,
                                                 # attention_weights_v, #[B,N_v,8,1,4]
                                                 # sampling_value_v, #[B*8,N_v*4,8]
                                                 )

        output_i = self.output_proj_i(output_i)#[B,N_v,64]
        output_dict['output_i'] = output_i
        return output_dict

    def KDTree_nearest_search(self, query_positon_v, value_v, q_lidar_indices):
        '''

        Args:
            query_positon_v: [B,N_v,8,2,3]
            value_v: [B,N_v,8,8]
            q_lidar_indices: [B,N_v,3]

        Returns:
            sampling_value: [B*8,N_v*4,8]


        '''
        B, N_v = q_lidar_indices.shape[:2]
        query_positon_v = query_positon_v.flatten(1,3)#[B,N_v*8*2,3]
        #[B,N_v,8,8]->[B,8,N_v,8]->[B*8,N_v,8]
        value_v = value_v.transpose(1, 2).flatten(0, 1)

        indice_list = []
        distance_list = []
        for b in range(B):
            # 建立cKDTree
            kdtree = cKDTree(q_lidar_indices[b].cpu().numpy())
            # nndescent = NNDescent(q_lidar_indices[b].cpu().numpy())
            # 指定空间位置
            query_position_b = query_positon_v[b].clone().detach().cpu().numpy()
            distance, index = kdtree.query(query_position_b, k=2)
            # distance, index = nndescent.query(query_position_b, k=2)
            indice = torch.tensor(index, device=query_positon_v.device)#[N_v*8*2,2]
            distance_t = torch.tensor(distance, device=query_positon_v.device)
            indice_list.append(indice)
            distance_list.append(distance_t)
        # [B,N_v*8*2,2] -> [B,N_v*8*2*2] -> [B*8,N_v*2*2]
        indices = torch.stack(indice_list, dim=0).flatten(-2).reshape(B*self.n_heads, N_v*4)
        distances = torch.stack(distance_list, dim=0).flatten(-2).reshape(B*self.n_heads, N_v*4)

        indices_mask = (distances <= 2).unsqueeze(-1) # [B*8,N_v*2*2,1]
        sampling_value_v = value_v[torch.arange(B*self.n_heads).unsqueeze(1), indices, :]#[B*8,N_v*4,8]
        sampling_value_v = sampling_value_v * indices_mask
        return sampling_value_v

    def KDTree_nearest_search_multiprocess(self, query_positon_v, value_v, q_lidar_indices):

        B, N_v = q_lidar_indices.shape[:2]
        query_positon_v = query_positon_v.flatten(1, 3)  # [B,N_v*8*2,3]
        # [B,N_v,8,8]->[B,8,N_v,8]->[B*8,N_v,8]
        value_v = value_v.transpose(1, 2).flatten(0, 1)
        q_lidar_indices_np = q_lidar_indices.detach().cpu().numpy()
        query_positon_v_np = query_positon_v.detach().cpu().numpy()




        # def perform_queries(b):
        #     # indice_list = []
        #     # distance_list = []
        #     neighbors_batch = []
        #
        #     kdtree = cKDTree(q_lidar_indices[b].cpu().numpy())
        #     # nndescent = NNDescent(q_lidar_indices[b].cpu().numpy())
        #     # 指定空间位置
        #     query_position_b = query_positon_v[b].clone().detach().cpu().numpy()
        #     distance, index = kdtree.query(query_position_b, k=2)
        #     # distance, index = nndescent.query(query_position_b, k=2)
        #     indice = torch.tensor(index, device=query_positon_v.device)  # [N_v*8*2,2]
        #     distance_t = torch.tensor(distance, device=query_positon_v.device)
        #     # indice_list.append(indice)
        #     # distance_list.append(distance_t)
        #     # neighbors_batch.append((indice, distance_t))
        #
        #     return indice, distance_t

        indice_list = []
        distance_list = []
        results_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
        # with Pool() as pool:
            results = list(executor.map(partial(perform_queries, q_lidar_indices=q_lidar_indices_np,
                                       query_positon_v=query_positon_v_np),range(B)))
            results_list.extend(results)

        for i in range(B):
            results_batch = results_list[i]
            indice_batch, distance_batch = results_batch[:2]
            indice_list.extend(torch.tensor(indice_batch, device="cuda:0"))
            distance_list.extend(torch.tensor(distance_batch, device="cuda:0"))


        indices = torch.stack(indice_list, dim=0).flatten(-2).reshape(B * self.n_heads, N_v * 4)
        distances = torch.stack(distance_list, dim=0).flatten(-2).reshape(B * self.n_heads, N_v * 4)

        indices_mask = (distances <= 2).unsqueeze(-1)  # [B*8,N_v*2*2,1]
        sampling_value_v = value_v[torch.arange(B * self.n_heads).unsqueeze(1), indices, :]  # [B*8,N_v*4,8]
        sampling_value_v = sampling_value_v * indices_mask
        return sampling_value_v

def perform_queries(b, q_lidar_indices, query_positon_v):
    # indice_list = []
    # distance_list = []
    neighbors_batch = []

    kdtree = cKDTree(q_lidar_indices[b])
    # nndescent = NNDescent(q_lidar_indices[b].cpu().numpy())
    # 指定空间位置
    query_position_b = query_positon_v[b]
    distance, index = kdtree.query(query_position_b, k=2)
    # distance, index = nndescent.query(query_position_b, k=2)
    # indice = torch.tensor(index, device="cuda:0")  # [N_v*8*2,2]
    # distance_t = torch.tensor(distance, device="cuda:0")


    return index, distance












