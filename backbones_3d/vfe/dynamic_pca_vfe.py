import torch
from torch_scatter import scatter_mean, scatter_add
from .vfe_template import VFETemplate

class DynamicPcaVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

    def get_output_feature_dim(self):
        return self.num_point_features + 5  # 增加PCA特征维度

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       point_coords[:, 0] * self.scale_yz + \
                       point_coords[:, 1] * self.scale_z + \
                       point_coords[:, 2]
        points_data = points[:, 1:].contiguous()

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = scatter_mean(points_data, unq_inv, dim=0)

        # 中心化点云
        mean_centered_points = points_data - points_mean[unq_inv]

        # 初始化协方差矩阵
        cov_matrix = torch.zeros((len(unq_coords), 4, 4), device=points.device)

        # 计算协方差矩阵
        valid_mask = unq_cnt > 1
        if valid_mask.any():
            valid_centered_points = mean_centered_points[valid_mask[unq_inv]]
            valid_unq_inv = unq_inv[valid_mask[unq_inv]]
            valid_cov_matrix = scatter_add(mean_centered_points.unsqueeze(-1) * mean_centered_points.unsqueeze(-2),
                                           unq_inv, dim=0)[valid_mask] #valid_cov_matrix.shape[0]为有效体素（num_points>1）个数
            valid_unq_cnt = unq_cnt[valid_mask].float()
            valid_cov_matrix = valid_cov_matrix / (valid_unq_cnt.view(-1, 1, 1) - 1)
            cov_matrix[valid_mask] = valid_cov_matrix

        # 计算特征值和特征向量
        eig_vals, eig_vecs = torch.linalg.eigh(cov_matrix.cpu())
        eig_vals = eig_vals.to(cov_matrix.device)
        eig_vecs = eig_vecs.to(cov_matrix.device)
        max_eig_vals, max_indices = torch.max(eig_vals, dim=1)
        first_principal_directions = eig_vecs[torch.arange(len(eig_vecs)), :, max_indices]

        # 处理单点体素
        single_point_mask = unq_cnt == 1
        single_point_pca_features = torch.cat(
            [points_mean[single_point_mask], torch.zeros((single_point_mask.sum(), 5), device=points.device),
             ], dim=1)

        # 合并PCA特征
        pca_features = torch.cat([points_mean, first_principal_directions, max_eig_vals.unsqueeze(-1)], dim=1)
        pca_features[single_point_mask] = single_point_pca_features

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['voxel_features'] = pca_features.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict