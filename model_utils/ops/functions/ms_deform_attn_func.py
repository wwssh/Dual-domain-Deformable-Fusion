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

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from .....ops.group_points.group_points import QueryAndGroup
import math

import MultiScaleDeformableAttention as MSDA



class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations,
                                attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1 # 把范围从[0,1]转换到[-1,1], F.grid_sample要求grid的范围是[-1,1]
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)#[16,8,94,311]
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)#[16,N_v,4,2]
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)#[16,8,N_v,4]
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    #更新的图像特征
    output1 = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)

    return output1.transpose(1, 2).contiguous()



def ms_deform_attn_core_pytorch_key_aware(query, value, key,  value_spatial_shapes, sampling_locations,
                                          query_proj, i_query_proj, i_query=None):
    #C_:CC_ / M ; L_:num levels ; P_:sampling point(4)
    B_, S_, M_, C_ = value.shape#(B,D*H*W,M,CC_/M)
    _, N_v, M_, L_, P_, _ = sampling_locations.shape #(B,N_v,M,L,P,2)
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    key_list = key.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    sampling_key_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B_*M_, C_, H_, W_)
        key_l_ = key_list[lid_].flatten(2).transpose(1, 2).reshape(B_*M_, C_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # B_*M_, C_, N_v, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

        # N_*M_, D_, Lq_, P_
        sampling_key_l__ = F.grid_sample(key_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_key_list.append(sampling_key_l__)

    # B_*M_, C_, N_v, P_ -> B_*M_, C_, N_v, L_, P_ -> B_*M_, C_, N_v, L_*P_
    key = torch.stack(sampling_key_list, dim=-2).flatten(-2)
    value = torch.stack(sampling_value_list, dim=-2).flatten(-2)

    # B_*M_, C_, N_v, L_*P_ -> B*M, N_v, L*P, C -> B*M*N_v, L*P, C
    key = key.permute(0, 2, 3, 1).flatten(0, 1)
    value = value.permute(0, 2, 3, 1).flatten(0, 1)

    B_, N_v, CC_ = query.shape
    query = query_proj(query)
    query = query.view(B_, N_v, M_, CC_ // M_)
    query = query.permute(0, 2, 1, 3).flatten(0, 2)  # B, N_v, M, C -> B, M, N_v, C -> B*M*N_v, C
    query = query.unsqueeze(-2)  # B*M*N_v, 1, C

    if i_query is not None:
        i_query = i_query_proj(i_query)
        i_query = i_query.view(B_, N_v, M_, CC_ // M_)
        i_query = i_query.permute(0, 2, 1, 3).flatten(0, 2)
        i_query = i_query.unsqueeze(-2)

        query = query + i_query

    dk = query.size()[-1]
    # B*M*N_v, 1, C, x B*M*N_v, L*P, C   ==  B*M*N_v, 1, L*P
    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    attention_weights = F.softmax(attention_weights, -1)

    # B*M*N_v, 1, L*P x B*M*N_v, L*P, C -> B*M*N_v, 1, C
    output = attention_weights.matmul(value)
    # B*M*N_v, 1, C ->B*M*N_v,C->B,M,N_v,C->B, N_v, M,  C
    output = output.squeeze(-2).view(B_, M_, N_v, C_).permute(0, 2, 1, 3)
    output = output.flatten(2) #B, N_v, CC_(M*C_)
    return output.contiguous()

def ms_deform_pool_core(query, key_proj, sampling_locations, attention_weights_h_list, ms_lidar_features):
    # assert sampling_locations.shape[3] == value_spatial_shapes.shape[0]
    # sampling_locations_true = sampling_locations * value_spatial_shapes[None, None, None, :, None, :] #B,200,8,3,4,2
    # sampling_locations_true = torch.round(sampling_locations_true)
    sampling_features_list = []
    B_, N_,M_,L_,P_,_ = sampling_locations.shape
    assert len(ms_lidar_features) == L_
    sampling_grids = 2 * sampling_locations - 1#B,200,8,3,4,2

    sampling_grids_bev = sampling_grids[:, :, :, 0, :, :] #B,200,8,4,2
    sampling_grids_bev = sampling_grids_bev.flatten(2,3)#B,200,8*4,2
    value_bev = ms_lidar_features[0]#B,128,200,176
    sampling_features = F.grid_sample(value_bev, sampling_grids_bev,  mode='bilinear', padding_mode='zeros',
                                      align_corners=False).permute(0, 2, 3, 1)#B，128，200，8*4 -> B,200,8*4, 128
    sampling_features_list.append(sampling_features)

    for l in range(L_-1):
        l += 1
        value_voxel = ms_lidar_features[l]#B,C,D,H,W
        _, C_, D_, _, _= value_voxel.shape
        value_voxel = value_voxel.transpose(1, 2).flatten(0, 1)#B*D,C,H,W
        sampling_grids_voxel = sampling_grids[:, :, :, l, :, :]  # B,200,8,4,2
        # B,D,200,8,4,2 -> B*D,200,8,4,2 -> B*D,200,8*4,2
        sampling_grids_voxel = sampling_grids_voxel.unsqueeze(1).repeat(1, D_, 1, 1, 1, 1).flatten(0, 1).flatten(2, 3)
        sampling_features_v = F.grid_sample(value_voxel, sampling_grids_voxel, mode='bilinear', padding_mode='zeros',
                                      align_corners=False).reshape(B_, D_, C_, N_, M_*P_) # B*D,C,200,8*4->B,D,C,200,8*4
        sampling_features_v = sampling_features_v.permute(0, 2, 3, 4, 1)#B,C,200,8*4,D
        sampling_features_v_maxpool = F.max_pool3d(sampling_features_v, kernel_size=(1,1,D_), stride=(1,1,D_)).squeeze(-1).permute(0, 2, 3, 1)
        # attn_weight_h = attention_weights_h_list[l - 1][:, None, :, None, :]#B,1,200,1,5
        # sampling_features_attnh = (sampling_features_v * attn_weight_h).sum(-1).permute(0, 2, 3, 1)#B,C,200,8*4->B,200,8*4,C
        sampling_features_list.append(sampling_features_v_maxpool)

    samp_feat_list = []
    for i, samp_feat in enumerate(sampling_features_list):
        samp_feat = key_proj[i](samp_feat)  #
        samp_feat_list.append(samp_feat)
    key= value = torch.cat(samp_feat_list, dim=2)  # B,200,8*4*3,128

    query = query[:, :, None, :]#B,200, 1,128
    dk = query.shape[-1]
    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    attention_weights = F.softmax(attention_weights, -1)

    output = attention_weights.matmul(value).squeeze(2) #B,200,1,128 -> B,200,128


    return output









    # N_, S_, M_, D_ = value.shape
    # _, Lq_, L_, M_, P_, _ = sampling_locations.shape #B,200,Level,head,point,2
    # value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # sampling_grids = 2 * sampling_locations - 1 # 把范围从[0,1]转换到[-1,1], F.grid_sample要求grid的范围是[-1,1]
    # sampling_value_list = []
    # for lid_, (H_, W_) in enumerate(value_spatial_shapes):
    #     # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
    #     value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)#[16,8,94,311]
    #     # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
    #     sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)#[16,N_v,4,2]
    #     # N_*M_, D_, Lq_, P_
    #     sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
    #                                       mode='bilinear', padding_mode='zeros', align_corners=False)#[16,8,N_v,4]
    #     sampling_value_list.append(sampling_value_l_)
    # # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # #更新的图像特征
    # output1 = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    #
    # return output1.transpose(1, 2).contiguous()