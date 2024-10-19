import torch
from torch import nn
import torch.nn.functional as F
from ..model_utils.ops.modules import BEVMSDeformAttn, MSDeformAttn
import os
from .attentions import attn_dict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed



    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, key_padding_mask=None, attn_mask=None):
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed),
                                     key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]



        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query


class DeformTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.bev_ms_attn = BEVMSDeformAttn(d_model=d_model, q_model=d_model, n_levels=3)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, batch_dict, query, bev_features, query_pos, key_pos, input_shape=None, key_padding_mask=None, attn_mask=None):
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        if input_shape is not None:
            input_shapes = torch.tensor(input_shape, device=query.device)
        else:
            # input_shapes = batch_dict['multi_scale_3d_features']['x_conv4'].spatial_shape[-2:]
            input_shapes = [[200, 176], [200, 176], [400, 352]]
            input_shapes = torch.tensor(input_shapes, device=query.device)

        x_conv4 = batch_dict['multi_scale_3d_features']['x_conv4']
        x_conv3 = batch_dict['multi_scale_3d_features']['x_conv3']
        bev = bev_features.clone()
        ms_lidar_features = [bev, x_conv4.dense(), x_conv3.dense()]


        query = query.permute(2, 0, 1)
        # key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        #如果要用多尺度bev特征输入，需要修改此处
        # if input_shapes.shape[0] == 1:
        #     input_level_start_index = torch.tensor([0], device=query.device)
        # else:
        #     nlevels = input_shapes.shape[0]
        #     input_level_start_index = torch.tensor([0], device=query.device)
        #     for i in range(nlevels - 1):
        #         x = torch.tensor([input_shapes[i, 0] * input_shapes[i, 1]], device=query.device)
        #         input_level_start_index = torch.cat((input_level_start_index, x), dim=0)


        query2 = self.bev_ms_attn(query=self.with_pos_embed(query, query_pos_embed).permute(1, 0, 2),
                                  reference_points=(query_pos / input_shapes[None, None, -1, :].flip(-1))[:, :, None, :],
                                  # input_flatten=key.permute(1, 0, 2),
                                  input_spatial_shapes=input_shapes,
                                  ms_lidar_features = ms_lidar_features
                                  )

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query


class DeformableTransformerFusionDecoderLayer(nn.Module):

    def __init__(self,
                 d_model=128,
                 q_model=128,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=1,
                 n_heads=8,
                 n_points=4,
                 ):
        super().__init__()



        # self attention
        self.q_method = 'sum'
        self.q_rep_place = 'weight'
        self.d_model = d_model
        self.self_attn = MSDeformAttn(d_model, q_model, n_levels, n_heads,
                                      n_points, q_method=self.q_method,
                                      q_rep_place=self.q_rep_place)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

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

        self.fusion_layer = attn_dict['BiGateSum1D_2'](q_model, q_model)
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

        # voxel attention

        # Fusion
        q_feat, q_i_feat = self.fusion_layer(q_feat, q_i_feat)

        # ffn
        q_i_feat = self.forward_i_ffn(q_i_feat)
        q_feat = self.forward_p_ffn(q_feat)

        return q_feat, q_i_feat



