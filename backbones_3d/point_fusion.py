

def point_fusion_mvx(self,x_list,
                     batch_dict,
                     img_dict,
                     voxel_stride=1):

    def construct_multimodal_features(x_list, x_rgb, batch_dict, fuse_sum=False):
        batch_index = x_list[-1].indices[:, 0]  # indices[N_v,4] (B,z,y,x)
        spatial_indices = x_list[-1].indices[:, 1:] * voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:
                                                                               3]
        calibs = batch_dict["calib"]
        batch_size = batch_dict["batch_size"]
        h, w = batch_dict["images"].shape[2:]

        x_rgb_int = nn.functional.interpolate(x_rgb[0], (h, w),
                                              mode="bilinear")  # 插值到图像原空间维度[B,256,h,w]

        image_with_voxelfeatures = []
        voxels_2d_int_list = []
        filter_idx_list = []


        for b in range(batch_size):
            x_rgb_batch = x_rgb[0][b]
            x_rgb_batch_int = x_rgb_int[b]  # 插值
            calib = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index == b]
            voxel_features_sparse = x_list[-1].features[batch_index == b]

            # Reverse the point cloud transformations to the original coords.
            if "noise_scale" in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict["noise_scale"][b]
            if "noise_rot" in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(
                    voxels_3d_batch[:, self.inv_idx].unsqueeze(0),
                    -batch_dict["noise_rot"][b].unsqueeze(0),
                )[0, :, self.inv_idx]
            if "flip_x" in batch_dict:
                voxels_3d_batch[:,
                1] *= -1 if batch_dict["flip_x"][b] else 1
            if "flip_y" in batch_dict:
                voxels_3d_batch[:,
                2] *= -1 if batch_dict["flip_y"][b] else 1

            voxels_2d, _ = calib.lidar_to_img(
                voxels_3d_batch[:, self.inv_idx].cpu().numpy())


            voxels_2d_int = torch.Tensor(voxels_2d).to(
                x_rgb_batch.device).long()

            filter_idx = ((0 <= voxels_2d_int[:, 1]) *
                          (voxels_2d_int[:, 1] < h) *
                          (0 <= voxels_2d_int[:, 0]) *
                          (voxels_2d_int[:, 0] < w))

            filter_idx_list.append(filter_idx)
            voxels_2d_int = voxels_2d_int[filter_idx]
            voxels_2d_int_list.append(voxels_2d_int)

            image_features_batch = torch.zeros(
                (voxel_features_sparse.shape[0], x_rgb_batch_int.shape[0]),
                device=x_rgb_batch_int.device,
            )  # [N_v,256]
            image_features_batch[
                filter_idx] = x_rgb_batch_int[:, voxels_2d_int[:, 1],
                              voxels_2d_int[:, 0]].permute(
                1, 0)

            if fuse_sum:
                image_with_voxelfeature = (image_features_batch +
                                           voxel_features_sparse)
            else:
                image_with_voxelfeature = torch.cat(
                    [image_features_batch, voxel_features_sparse],
                    dim=1)
            image_with_voxelfeatures.append(image_with_voxelfeature)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures)
        return image_with_voxelfeatures


    x_rgb = []
    for key in img_dict:
        x_rgb.append(img_dict[key])
    features_multimodal = construct_multimodal_features(
        x_list, x_rgb, batch_dict, True)
    x_mm = spconv.SparseConvTensor(features_multimodal, x_list[-1].indices,
                                   x_list[-1].spatial_shape, x_list[-1].batch_size)
    return x_mm






def point_fusion(self,
                     x_list,
                     batch_dict,
                     img_dict,
                     fusion_method,
                     Dual_spatial=None,
                     voxel_stride=1):
    def construct_multimodal_features(x_list,
                                      x_rgb,
                                      batch_dict,
                                      Dual_spatial=None,
                                      fuse_sum=False):
        """
        Construct the multimodal features with both lidar sparse features and image features.
        Args:
            x: [N, C] lidar sparse features
            x_rgb: [b, c, h, w] image features
            batch_dict: input and output information during forward
            fuse_sum: bool, manner for fusion, True - sum, False - concat

        Return:
            image_with_voxelfeatures: [N, C] fused multimodal features
        """
        voxel_indices = x_list[-1].indices[:, 1:]
        batch_index = x_list[-1].indices[:, 0]  # indices[N_v,4] (B,z,y,x)
        spatial_indices = x_list[-1].indices[:, 1:] * voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:
                                                                               3]
        calibs = batch_dict["calib"]
        batch_size = batch_dict["batch_size"]
        h, w = batch_dict["images"].shape[2:]
        D, H, W = (self.point_cloud_range[3:] - self.point_cloud_range[:3]).cpu().numpy()

        x_rgb_int = nn.functional.interpolate(x_rgb[0], (h, w),
                                              mode="bilinear")  # 插值到图像原空间维度[B,256,h,w]

        voxels_2d_int_list = []
        filter_idx_list = []
        pts_list = []  # 体素空间位置（z,y,x)
        coor_2d_list = []  # voxel_2d_norm
        coor_3d_list = []  # voxel_3d_norm
        pts_feats_list = []  # 体素稀疏特征
        pts_img_feats_list = []
        num_points = []
        pts_indices_list = []

        for b in range(batch_size):
            x_rgb_batch = x_rgb[0][b]
            x_rgb_batch_int = x_rgb_int[b]  # 插值

            calib = calibs[b]

            voxels_3d_batch = voxels_3d[batch_index == b]
            #  voxels_3d_batch = batch_dict['points'][batch_dict['points'][:, 0] == b][:, [3, 2, 1]]
            voxel_features_sparse = x_list[-1].features[batch_index == b]
            num_points.append(voxel_features_sparse.shape[0])

            if Dual_spatial:
                voxel_indices_batch = voxel_indices[batch_index == b]
                voxels_3d_norm = (voxels_3d_batch - self.point_cloud_range[:3]).cpu().numpy() / np.array(
                    [D, H, W])  # (z,y,x)
                voxels_3d_norm = voxels_3d_norm[:, [2, 1, 0]]  # ()xyz

            # Reverse the point cloud transformations to the original coords.
            if "noise_scale" in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict["noise_scale"][b]
            if "noise_rot" in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(
                    voxels_3d_batch[:, self.inv_idx].unsqueeze(0),
                    -batch_dict["noise_rot"][b].unsqueeze(0),
                )[0, :, self.inv_idx]
            if "flip_x" in batch_dict:
                voxels_3d_batch[:,
                1] *= -1 if batch_dict["flip_x"][b] else 1
            if "flip_y" in batch_dict:
                voxels_3d_batch[:,
                2] *= -1 if batch_dict["flip_y"][b] else 1

            voxels_2d, _ = calib.lidar_to_img(
                voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_norm = voxels_2d / np.array([w, h])

            voxels_2d_int = torch.Tensor(voxels_2d).to(
                x_rgb_batch.device).long()

            filter_idx = ((0 <= voxels_2d_int[:, 1]) *
                          (voxels_2d_int[:, 1] < h) *
                          (0 <= voxels_2d_int[:, 0]) *
                          (voxels_2d_int[:, 0] < w))

            filter_idx_list.append(filter_idx)
            voxels_2d_int = voxels_2d_int[filter_idx]
            voxels_2d_int_list.append(voxels_2d_int)

            image_features_batch = torch.zeros(
                (voxel_features_sparse.shape[0], x_rgb_batch_int.shape[0]),
                device=x_rgb_batch_int.device,
            )  # [N_v,256]
            image_features_batch[
                filter_idx] = x_rgb_batch_int[:, voxels_2d_int[:, 1],
                              voxels_2d_int[:, 0]].permute(
                1, 0)

            pts_img_feats_list.append(image_features_batch)

            coor_2d_list.append(voxels_2d_norm)
            pts_list.append(voxels_3d_batch)

            pts_feats_list.append(voxel_features_sparse)
            if Dual_spatial:
                pts_indices_list.append(voxel_indices_batch)
                coor_3d_list.append(voxels_3d_norm)


        n_max = 0
        pts_feats_b = torch.zeros((batch_size, self.max_num_nev,
                                       x_list[-1].features.shape[1])).cuda()
        i_channel = sum(list([a.shape[1] for a in x_rgb]))
        pts_i_feats_b = torch.zeros(
                (batch_size, self.max_num_nev, i_channel)).cuda()
        coor_2d_b = torch.zeros(
                (batch_size, self.max_num_nev, 2)).cuda()
        coor_3d_b = torch.zeros(
                (batch_size, self.max_num_nev, 3)).cuda()
        indices_b = torch.zeros(
                (batch_size, self.max_num_nev, 3)).cuda()

        pts_b = torch.zeros((batch_size, self.max_num_nev, 3)).cuda()
        for b in range(batch_size):

            pts_b[b, :pts_list[b].shape[0]] = pts_list[b]
            coor_2d_b[b, :pts_list[b].shape[0]] = torch.tensor(
                    coor_2d_list[b]).cuda()
            coor_3d_b[b, :pts_list[b].shape[0]] = torch.tensor(
                    coor_3d_list[b]).cuda()
            indices_b[b, :pts_list[b].shape[0]] = torch.tensor(
                    pts_indices_list[b]).cuda()

            n_max = max(n_max, pts_list[b].shape[0])
            pts_feats_b[b, :pts_list[b].shape[0]] = pts_feats_list[b]
            pts_i_feats_b[b, :pts_list[b].shape[0]] = pts_img_feats_list[b]
        if self.attention:
            x_rgb = self.ifat(
                    x_rgb=x_rgb, x_list=x_list, batch_dict=batch_dict
                )
        enh_feat = self.actr(
                v_feat=pts_feats_b[:, :n_max],
                v_i_feat=pts_i_feats_b[:, :n_max],
                grid=coor_2d_b[:, :n_max],
                grid_ld=coor_3d_b[:, :n_max],
                i_feats=x_rgb,
                lidar_grid=pts_b[:, :n_max, self.inv_idx],
                v_indices=indices_b[:, :n_max, self.inv_idx],
                Dual_spatial=Dual_spatial
            )
        enh_feat_cat = torch.cat(
                [f[:np] for f, np in zip(enh_feat, num_points)])
        if fuse_sum:
            enh_feat_cat = enh_feat_cat + x_list[-1].features
        else:
            enh_feat_cat = torch.cat([enh_feat_cat, x_list[-1].features], dim=1)
        return enh_feat_cat

    x_rgb = []
    for key in img_dict:
        x_rgb.append(img_dict[key])
    features_multimodal = construct_multimodal_features(
        x_list, x_rgb, batch_dict, Dual_spatial, True)
    x_mm = spconv.SparseConvTensor(features_multimodal, x_list[-1].indices,
                                   x_list[-1].spatial_shape, x_list[-1].batch_size)
    return x_mm