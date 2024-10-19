import torch
import torch.nn.functional as F
from torch import Tensor, nn
import copy
from torch.nn.init import constant_, normal_, uniform_, xavier_uniform_
from pcdet.models.model_utils.position_encoding import PositionEmbeddingSineD

from .ops.modules import MSDeformVoxelAttn



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DeformableVoxelTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=64,
                 d_ffn=1024,
                 deform_head=8,
                 n_points=4,
                 n_levels=5,
                 dropout=0.1,
                 q_method=None,
                 q_rep_place=None
                 ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.deform_head = deform_head
        self.deform_attn = MSDeformVoxelAttn(d_model, n_levels, deform_head,
                                             n_points, q_method=q_method, q_rep_place=q_rep_place)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                q_feat,
                dense_voxel_flatten,
                reference_points,
                spatial_shapes,
                level_start_index,
                q_pos=None,
                q_i_feat=None,
                ):
        '''

        Args:
            q_feat: (B,N_v,C)
            dense_voxel_flatten: (B,D*H*W,C)
            reference_points: (B, N_v, 2), range in [0, 1],(x,y) top-left (0,0), bottom-right (1, 1)
            spatial_shapes: (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1=4}, W_{L-1=4})]
            level_start_index: (n_levels, ), [0, H*W,2*H*W,3*H*W,4*H*W ]
            q_pos: (B,N_v,C)

        Returns:

        '''
        if q_i_feat is not None:
            src2 = self.deform_attn(self.with_pos_embed(q_feat, q_pos), reference_points,
                                    dense_voxel_flatten, spatial_shapes, level_start_index,
                                    i_query=self.with_pos_embed(q_i_feat, q_pos))
        else:
            src2 = self.deform_attn(self.with_pos_embed(q_feat, q_pos), reference_points,
                                    dense_voxel_flatten, spatial_shapes, level_start_index,
                                   )
        q_feat = q_feat + self.dropout1(src2)
        q_feat = self.norm1(q_feat)

        q_feat = self.forward_ffn(q_feat)

        return q_feat


class DeformableVoxelTransformerEncoder(nn.Module):
    def __init__(self,
                 lt_cfg):
        super().__init__()
        num_layers = lt_cfg.num_layers
        d_model = lt_cfg.d_model
        self.layers = _get_clones(
            DeformableVoxelTransformerEncoderLayer(
                d_model=lt_cfg.d_model,
                deform_head=lt_cfg.nhead,
                n_points=lt_cfg.n_points,
                n_levels=lt_cfg.n_levels,
                dropout=lt_cfg.dropout,
                q_method=lt_cfg.q_method,
                q_rep_place=lt_cfg.q_rep_place),
            num_layers)
        # self.pos_embed = nn.Sequential(
        #     nn.Linear(3, d_model // 2),
        #     nn.Linear(d_model // 2, d_model),
        # )

    def forward(self, q_feat, dense_voxel_flatten, reference_points,
                spatial_shapes, level_start_index, q_pos, q_i_feat=None):
        '''

        Args:
            q_feat:
            dense_voxel_flatten:(B,D*H*W,C)
            reference_points: (B,N_v,2)(HW)
            spatial_shapes: [(H_0, W_0), (H_1, W_1), ..., (H_{L-1=4}, W_{L-1=4})]
            level_start_index: [0, H*W,2*H*W,3*H*W,4*H*W ]
            q_pos: (B,N_v,C)

        Returns:

        '''
        # q_pos = self.pos_embed(reference_points_3d)
        # reference_points = reference_points_3d[..., 1:] #(B,N_v,2)(HW)

        for layer in self.layers:
            q_feat = layer(
                q_feat,
                dense_voxel_flatten,
                reference_points,
                spatial_shapes,
                level_start_index,
                q_pos,
                q_i_feat,
            )
        return q_feat


class BEVAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, voxel_dense_flatten_feat, D_pos):
        '''

        Args:
            voxel_dense_flatten_feat: (B*H*W,D,C)
            D_pos: (D,C)

        Returns:

        '''
        src = voxel_dense_flatten_feat
        D_pos = D_pos[None, ...]#(1,D,C)
        q = k = voxel_dense_flatten_feat + D_pos
        v = voxel_dense_flatten_feat
        src2, _ = self.self_attn(q, k, v)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class BEVAttnModule(nn.Module):
    def __init__(self, d_model, nhead, num_feat_levels, num_layers=1, dropout=0.0):
        super().__init__()

        # self.level_embed = nn.Parameter(
        #     torch.Tensor(num_feat_levels, d_model))
        bevattnlayer = BEVAttnLayer(d_model, nhead, dropout)
        self.BEVAttn = _get_clones(bevattnlayer, num_layers)
        # self.BEVAttn = BEVAttnLayer(d_model, nhead, dropout)
        # self._reset_parameters()
        self.pe_D = PositionEmbeddingSineD(64, normalize=True)

    # def _reset_parameters(self):
    #     normal_(self.level_embed)

    def forward(self, voxel_dense_feat, indices):
        '''

        Args:
            voxel_dense_feat: (B,C,D,H,W)
            indices: (B,N_v,3)(zyx)or(DHW)


        Returns:

        '''
        B, C, D, H, W = voxel_dense_feat.shape
        indices = indices.long()
        #(B*H*W,D,C)
        voxel_dense_flatten_feat = voxel_dense_feat.permute(0, 3, 4, 2, 1).flatten(0, 2)
        pos_D = self.pe_D(voxel_dense_flatten_feat)
        #self.level_embed和pos_D二选一
        for layer in self.BEVAttn:
            voxel_dense_flatten_feat = layer(voxel_dense_flatten_feat, pos_D)
        output_dense_feat = voxel_dense_flatten_feat.view(B, H, W, D, C).permute(0, 4, 3, 1, 2)#(B,C,D,H,W)

        output_sparse_feat = output_dense_feat[torch.arange(B).unsqueeze(-1), :, indices[:, :, 0],
                             indices[:, :, 1], indices[:, :, 2]]#(B,N_v,C)
        return output_dense_feat, output_sparse_feat
