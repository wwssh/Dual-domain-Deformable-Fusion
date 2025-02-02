# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, uniform_, xavier_uniform_
# from pcdet.models.model_utils.pointformer import LocalTransformer
from pcdet.models.model_utils.voxel_transformer import DeformableVoxelTransformerEncoder, BEVAttnModule

from .ops.modules import MSDeformAttn
from .ops.modules import MSDeformVoxelAttn
from .actr_utils import inverse_sigmoid
from .attentions import attn_dict

import pickle


class DeformableTransformerACTR(nn.Module):

    def __init__(
        self,
        d_model=256,
        query_num_feat=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        model_name='ACTR',
        lt_cfg=None,
        feature_modal='lidar',
        hybrid_cfg=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.q_model = query_num_feat
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.model_name = model_name

        self.feature_modal = feature_modal
        if model_name == 'ACTRv2':
            self.BEVAttn = BEVAttnModule(self.d_model, nhead, lt_cfg.n_levels, num_layers=1, dropout=0.0)

        if feature_modal in ['hybrid']:
            encoder_layer = DeformableTransformerFusionEncoderLayer(
                self.d_model, self.q_model, dim_feedforward, dropout, activation,
                num_feature_levels, nhead, enc_n_points, hybrid_cfg)
        else:
            encoder_layer = DeformableTransformerEncoderLayer(
                self.d_model, self.q_model, dim_feedforward, dropout, activation,
                num_feature_levels, nhead, enc_n_points, hybrid_cfg)
        self.encoder = DeformableTransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            model_name=model_name,
            lt_cfg=lt_cfg)

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))
        self.voxel_level_embed = nn.Parameter(
            torch.Tensor(lt_cfg.n_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        for m in self.modules():
            if isinstance(m, MSDeformVoxelAttn):
                m._reset_parameters()
        normal_(self.level_embed)
        normal_(self.voxel_level_embed)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(
                N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def voxel_level_embedding(self, level_embed, voxel_pos, voxel_sparse_indices):
        '''

        Args:
            level_embed: (lvl,C)
            voxel_pos: (B,N_v,C)
            voxel_sparse_indices: (B,N_v,3)

        Returns:

        '''
        B, N_v, C = voxel_pos.shape
        lvl_indices = voxel_sparse_indices[..., 0].flatten(0, 1) #(B*N_v)
        slc_level_embed = level_embed[lvl_indices.long()].view(B, N_v, C) #(B,N_v,C)
        return voxel_pos + slc_level_embed

    def forward(self,
                srcs,
                masks,
                pos_embeds,
                q_feat_flatten,
                q_pos,
                q_ref_coors,
                q_lidar_grid=None,
                q_i_feat_flatten=None,

                voxel_dense_feat=None,
                voxel_sparse_indices=None,
                voxel_grid=None,
                voxel_pos=None
                ):
        # prepare input for encoder
        voxel_pos = self.voxel_level_embedding(self.voxel_level_embed, voxel_pos, voxel_sparse_indices)
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        dense_vfeat = voxel_dense_feat
        sparse_vfeat = q_feat_flatten
        if self.model_name == 'ACTRv2':
            #BEVAttention
            dense_vfeat, sparse_vfeat = self.BEVAttn(voxel_dense_feat, voxel_sparse_indices)
        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
            q_pos=q_pos,
            q_feat=q_feat_flatten,
            q_reference_points=q_ref_coors,
            q_lidar_grid=q_lidar_grid,
            q_i_feat=q_i_feat_flatten,

            dense_vfeat=dense_vfeat,
            sparse_vfeat=sparse_vfeat,
            voxel_grid=voxel_grid,
            voxel_pos=voxel_pos
        )

        return memory


class DeformableTransformerIACTR(nn.Module):

    def __init__(
        self,
        d_model=256,
        query_num_feat=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        model_name='IACTR',
        lt_cfg=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.q_model = query_num_feat
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.model_name = model_name

        encoder_layer = DeformableTransformerEncoderLayer(
            self.d_model, self.q_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                    num_encoder_layers)

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(
                N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self,
                srcs,
                masks,
                pos_embeds,
                q_feats,
                q_poss,
                q_ref_coors=None):
        # prepare input for encoder
        batch_size = srcs[0].shape[0]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        lvl_q_pos_embed_flatten = []
        q_feat_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed, q_feat, q_pos) in enumerate(
                zip(srcs, masks, pos_embeds, q_feats, q_poss)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            q_feat = q_feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            q_pos = q_pos.flatten(2).transpose(1, 2)
            lvl_q_pos_embed = q_pos + self.level_embed[lvl].view(1, 1, -1)
            lvl_q_pos_embed_flatten.append(lvl_q_pos_embed)
            src_flatten.append(src)
            q_feat_flatten.append(q_feat)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        q_feat_flatten = torch.cat(q_feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        lvl_q_pos_embed_flatten = torch.cat(lvl_q_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        enh_src = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
            q_pos=lvl_q_pos_embed_flatten,
            q_feat=q_feat_flatten,
            q_reference_points=q_ref_coors,
        )
        if q_ref_coors is not None:
            return enh_src

        enh_src_list = []
        level_start_index = torch.cat(
            [level_start_index,
             torch.tensor([src_flatten.shape[1]]).cuda()])
        for idx in range(level_start_index.shape[0] - 1):
            slice_enh_src = enh_src[:, level_start_index[idx]:
                                    level_start_index[idx + 1]]
            slice_enh_src = slice_enh_src.reshape(batch_size,
                                                  spatial_shapes[idx][0],
                                                  spatial_shapes[idx][1], -1)
            enh_src_list.append(slice_enh_src.permute(0, 3, 1, 2))

        return enh_src_list


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model=256,
                 q_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 hybrid_cfg=None,
                 ):
        super().__init__()

        # self attention
        self.d_model = d_model
        self.self_attn = MSDeformAttn(d_model, q_model, n_levels, n_heads,
                                      n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask=None,
                q_pos=None,
                q_feat=None,
                q_i_feat=None,
                ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(q_feat, q_pos), reference_points, src,
            spatial_shapes, level_start_index, padding_mask)
        q_feat = q_feat + self.dropout1(src2)
        q_feat = self.norm1(q_feat)

        # ffn
        q_feat = self.forward_ffn(q_feat)

        return q_feat, q_i_feat

class DeformableTransformerFusionEncoderLayer(nn.Module):

    def __init__(self,
                 d_model=256,
                 q_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 hybrid_cfg=None,
                 ):
        super().__init__()

        self.attn_layer = hybrid_cfg['attn_layer']
        self.q_method = hybrid_cfg.get('q_method', None)
        self.q_rep_place = hybrid_cfg.get('q_rep_place', None)

        # self attention
        self.d_model = d_model
        self.self_attn = MSDeformAttn(d_model, q_model, n_levels, n_heads,
                                      n_points, q_method=self.q_method,
                                      q_rep_place=self.q_rep_place)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # i_ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # p_ffn
        self.linear3 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout4 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(d_ffn, d_model)
        self.dropout5 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.fusion_layer = attn_dict[self.attn_layer](q_model, q_model)
        # self.fusion_layer = attn_dict[self.attn_layer](q_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_i_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward_p_ffn(self, src):
        src2 = self.linear4(self.dropout4(self.activation(self.linear3(src))))
        src = src + self.dropout5(src2)
        src = self.norm3(src)
        return src

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask=None,
                q_pos=None,
                q_feat=None,
                q_i_feat=None,
                ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(q_feat, q_pos), reference_points, src,
            spatial_shapes, level_start_index, padding_mask,
            i_query=self.with_pos_embed(q_i_feat, q_pos)
            )
        q_i_feat = q_i_feat + self.dropout1(src2)
        q_i_feat = self.norm1(q_i_feat)

        #voxel attention
        
        # Fusion
        q_feat, q_i_feat = self.fusion_layer(q_feat, q_i_feat)

        # ffn
        q_i_feat = self.forward_i_ffn(q_i_feat)
        q_feat = self.forward_p_ffn(q_feat)

        return q_feat, q_i_feat


class DeformableTransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 model_name='ACTR',
                 lt_cfg=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.model_name = model_name
        if model_name == 'ACTRv2':
            # self.lidar_attns = _get_clones(
            #     LocalTransformer(
            #         lt_cfg.npoint,
            #         lt_cfg.radius,
            #         lt_cfg.nsample,
            #         encoder_layer.d_model,
            #         encoder_layer.d_model,
            #         num_layers=lt_cfg.num_layers,
            #         attn_feat_agg_method=lt_cfg.get('attn_feat_agg_method',
            #                                         'unique'),
            #         feat_agg_method=lt_cfg.get('feat_agg_method', 'replace')),
            #     num_layers)
            self.voxel_attns = _get_clones(
                DeformableVoxelTransformerEncoder(lt_cfg), num_layers
            )


    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                pos=None,
                padding_mask=None,
                q_feat=None,
                q_pos=None,
                q_reference_points=None,
                q_lidar_grid=None,
                q_i_feat=None,

                dense_vfeat=None,
                sparse_vfeat=None,
                voxel_grid=None,
                voxel_pos=None
                ):
        output = src
        if q_reference_points is None:
            # for IACTR
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device)
        else:
            # for ACTR
            reference_points = q_reference_points[:, :,
                                                  None] * valid_ratios[:, None]
        if self.model_name == 'ACTRv2':
            B_, C_, D_, H_, W_ = dense_vfeat.shape
            dense_voxel_flatten = dense_vfeat.permute(0, 2, 3, 4, 1).flatten(1, 3)
            voxel_spatial_shapes = torch.zeros((D_, 2), device=reference_points.device)

            voxel_spatial_shapes = voxel_spatial_shapes + torch.tensor([H_, W_],device=voxel_spatial_shapes.device)

            voxel_level_start_index = torch.tensor([0, H_*W_, 2*H_*W_, 3*H_*W_, 4*H_*W_]).cuda()
            q_feat = sparse_vfeat

        # vislist = []

        for idx, layer in enumerate(self.layers):
            if self.model_name == 'ACTRv2':
                q_feat = self.voxel_attns[idx](
                                 q_feat,
                                 dense_voxel_flatten,
                                 voxel_grid[...,1:],
                                 voxel_spatial_shapes,
                                 voxel_level_start_index,
                                 voxel_pos,#(HW+D)
                                 q_i_feat,
                                 )
            q_feat, q_i_feat = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
                q_pos=q_pos,
                q_feat=q_feat,
                q_i_feat=q_i_feat,
            )
            # vislist.append(visdict)

        # with open('/home/wsh/research_project/OpenPCDet2-master/vis/vis.pkl', 'wb') as f:
        #     pickle.dump(vislist, f)

        return q_feat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(args, model_name='ACTR', lt_cfg=None):
    if 'ACTR' in model_name and 'IACTR' not in model_name:
        model_class = DeformableTransformerACTR
    elif 'IACTR' in model_name:
        model_class = DeformableTransformerIACTR
    return model_class(
        d_model=args.hidden_dim,
        query_num_feat=args.query_num_feat,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        model_name=model_name,
        lt_cfg=lt_cfg,
        feature_modal=args.feature_modal,
        hybrid_cfg=args.hybrid_cfg
    )
