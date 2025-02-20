#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from utils.logging_utils import Log
from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View
# from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, config, nr_objects, device):
        #
        # self.active_sh_degree = 0
        # self.max_sh_degree = config.sh_degree
        self.device = device
        #
        self._xyz = torch.empty(0, device=device)
        self._features_dc = torch.empty(0, device=device)
        # self._features_rest = torch.empty(0, device=device)
        self._scaling = torch.empty(0, device=device)
        self._rotation = torch.empty(0, device=device)
        self._opacity = torch.empty(0, device=device)
        self._obj_prob = torch.empty(0, device=device)
        self.max_radii_2d = torch.empty(0, device=device)
        self.xyz_gradient_accum = torch.empty(0, device=device)

        self.kf_idx = torch.empty(0, device="cpu").int()
        self.nr_obs = torch.empty(0, device="cpu").int()

        #
        self.nr_objects = nr_objects

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.obj_id_activation = torch.nn.functional.softmax

        self.rotation_activation = torch.nn.functional.normalize

        # self.config = config
        self.ply_input = None

        # TODO: make hyperparameter
        self.isotropic = config.isotropic

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        # features_dc = features_dc.unsqueeze(1)
        # features_rest = self._features_rest
        # return torch.cat((features_dc, features_rest), dim=1)
        return features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_obj_prob(self):
        return self.obj_id_activation(self._obj_prob)

    # def get_covariance(self, scaling_modifier=1):
    #     return self.covariance_activation(
    #         self.get_scaling, scaling_modifier, self._rotation
    #     )

    # def oneupSHdegree(self):
    #     if self.active_sh_degree < self.max_sh_degree:
    #         self.active_sh_degree += 1

    def create_viewpoint_pcd(
        self,
        viewpoint,
        cam_intrinsics,
        render_depth=None,
        render_opacity=None,
        init=False,
    ):
        #
        Log("Creating PCD from image", tag="Mapper")

        # apply exposure (learned during tracking) to gt_rgb
        if init:
            gt_rgb = viewpoint.rgb
        else:
            image_ab = (
                torch.exp(viewpoint.exposure_a)
            ) * viewpoint.rgb + viewpoint.exposure_b
            image_ab = torch.clamp(image_ab, 0.0, 1.0)
            gt_rgb = image_ab
        gt_rgb = gt_rgb.permute(1, 2, 0)  # (H, W, 3)

        # get gt_depth and gt_segments
        gt_depth = viewpoint.depth.unsqueeze(-1)  # (H, W, 1)
        gt_segments = viewpoint.segmentation.unsqueeze(-1)  # (H, W, 1)

        # 
        if not init:
            render_opacity = render_opacity.permute(2, 1, 0)  # (1, H, W) -> (H, W, 1)
            render_depth = render_depth.permute(2, 1, 0)  # (1, H, W) -> (H, W, 1)
            render_depth = render_depth.flatten()  # (H*W,)
            render_opacity = render_opacity.flatten()  # (H*W,)
        
        K = torch.tensor(
            [
                [cam_intrinsics.fx, 0, cam_intrinsics.cx],
                [0, cam_intrinsics.fy, cam_intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        K_inv = torch.linalg.inv(K)
        w2c = getWorld2View(viewpoint.R, viewpoint.T)
        c2w = torch.linalg.inv(w2c)

        if init:
            # Downsample factor for initialization
            downsample_factor = 32  # self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = 64  # self.config["Dataset"]["pcd_downsample"]
        # print("downsample_factor", downsample_factor)
        point_size = 0.01  # self.config["Dataset"]["point_size"]
        # if "adaptive_pointsize" in self.config["Dataset"]:
        # print("using adaptive point size")
        # if self.config["Dataset"]["adaptive_pointsize"]:
        # point_size = min(0.05, point_size * np.median(gt_depth))
        point_size = min(0.05, point_size * torch.median(gt_depth))

        # invert H and W
        gt_rgb = gt_rgb.permute(1, 0, 2)  # (W, H, 3)
        gt_depth = gt_depth.permute(1, 0, 2)  # (W, H, 1)
        gt_segments = gt_segments.permute(1, 0, 2)  # (W, H, 1)
        # flatten depth image
        points_depth = gt_depth.flatten()  # (H*W,)
        points_ids = gt_segments.flatten()  # (H*W,)
        points_rgb = gt_rgb.reshape(-1, 3)  # (H*W, 3)

        # densification mask to determine which pixels should be densified
        
        # depth valid (most important)
        mask = points_depth >= 1e-3
        
        if not init:
            # map isn’t adequately dense
            # (O(p) < 0.5)
            if render_opacity is not None:
                render_opacity_mask = render_opacity < 0.5
            else:
                render_opacity_mask = torch.ones_like(points_depth, dtype=torch.bool)
            
            # OR there should be new geometry in front of
            # the current estimated geometry (i.e., the ground-truth depth
            # is in front of the predicted depth
            # (D_GT(p) < D(p))

            condition_1 = points_depth < render_depth

            # AND
            # the depth error is greater than λ times the median depth error
            # λ = 50
            # (L1 D(p) > λ MDE)
            
            # get median depth error
            median_depth_error = torch.abs(points_depth - render_depth).median()
            
            condition_2 = torch.abs(points_depth - render_depth) > 50 * median_depth_error
            
            # combine conditions
            condition = torch.logical_and(condition_1, condition_2)
            condition = torch.logical_or(render_opacity_mask, condition)
            condition = torch.logical_and(mask, condition)
            mask = condition

        # get pixel coordinates
        width, height = gt_rgb.shape[0], gt_rgb.shape[1]
        pixels_x, pixels_y = torch.meshgrid(
            torch.arange(width, device=self.device),
            torch.arange(height, device=self.device),
            indexing="ij",
        )  # (W, H, 2)
        pixels = torch.stack([pixels_x, pixels_y], dim=-1).type(torch.int32)
        # get pixels centers
        points_2d_screen = pixels.float()  # cast to float32
        points_2d_screen = points_2d_screen + 0.5  # pixels centers
        points_2d_screen = points_2d_screen.reshape(-1, 2)  # (H*W, 2)
        # filtering
        points_2d_screen = points_2d_screen[mask]
        points_depth = points_depth[mask]
        points_rgb = points_rgb[mask]
        points_ids = points_ids[mask]
        
        # downsample
        keep_fraction_points = 1.0 / downsample_factor
        # get current number of points
        num_points = points_2d_screen.shape[0]
        # find nr point to keep
        num_points_to_keep = int(num_points * keep_fraction_points)
        # get random indices
        random_indices = torch.randperm(num_points)[:num_points_to_keep]
        # keep only the points with the random indices
        points_2d_screen = points_2d_screen[random_indices]
        points_depth = points_depth[random_indices]
        points_rgb = points_rgb[random_indices]
        points_ids = points_ids[random_indices]

        # unproject 2D screen points to camera space
        ones = torch.ones(
            (points_2d_screen.shape[0], 1),
            dtype=points_2d_screen.dtype,
            device=points_2d_screen.device,
        )
        augmented_points_2d_screen = torch.cat((points_2d_screen, ones), dim=1)
        augmented_points_2d_screen = augmented_points_2d_screen[..., None]  # (N, 3, 1)
        points_3d_camera = K_inv @ augmented_points_2d_screen  # (N, 3, 3) @ (N, 3, 1)
        # reshape to (N, 3)
        points_3d_camera = points_3d_camera.squeeze(-1)  # (N, 3)
        # multiply by depth
        points_3d_camera *= points_depth[..., None]
        # Transform points from camera space to world space
        # Rotate
        points_3d_world = (c2w[:3, :3] @ points_3d_camera.T).T
        # Add translation
        points_3d_world += c2w[:3, -1][None, ...]  # (1, 3)
        
        # convert to Gaussians
        # means
        points_3d = points_3d_world
        # features
        # fused_color = RGB2SH(points_rgb)
        features = points_rgb
        
        # fused_color = points_rgb
        # features = (
        #     torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
        #     .float()
        #     .to(self.device)
        # )
        # features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0
        
        # scales
        # TODO: compute dist2 with all existing points
        dist2 = (
            torch.clamp_min(
                distCUDA2(
                    points_3d_world
                ),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)
        # rotations
        rots = torch.zeros(
            (points_3d.shape[0], 4), device=self.device, dtype=torch.float32
        )
        rots[:, 0] = 1
        # opacities
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (points_3d.shape[0], 1), dtype=torch.float32, device=self.device
            )
        )

        return points_3d, features, scales, rots, opacities, points_ids

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd_seq(
        self,
        viewpoint,
        cam_intrinsics,
        kf_idx,
        render_depth=None,
        render_opacity=None,
        init=False,
    ):
        # assert depth and segmentation are torch tensors on same
        # device as rgb

        # assert isinstance(depth, torch.Tensor), "Depth must be a torch tensor"
        # assert isinstance(
        #     segmentation, torch.Tensor
        # ), "Segmentation must be a torch tensor"
        # assert depth.device == rgb.device, "Depth and RGB must be on same device"
        # assert (
        #     segmentation.device == rgb.device
        # ), "Segmentation and RGB must be on same device"

        # create pcd
        points_3d, features, scales, rots, opacities, object_id = (
            self.create_viewpoint_pcd(
                viewpoint,
                cam_intrinsics,
                render_depth=render_depth,
                render_opacity=render_opacity,
                init=init,
            )
        )
        #
        Log(f"Extending Map with {points_3d.shape[0]} Gaussians", tag="Mapper")

        new_xyz = nn.Parameter(points_3d.requires_grad_(True))
        
        new_features_dc = nn.Parameter(
            features.requires_grad_(True)
        )
        # new_features_dc = nn.Parameter(
        #     features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        # )
        # new_features_rest = nn.Parameter(
        #     features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        # )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        # object id is an int from semantic segmentation
        # convert it to a tensor (N, nr_objects) with 1 at the object_id
        new_obj_prob = torch.zeros(
            (new_xyz.shape[0], self.nr_objects), dtype=torch.float32, device=self.device
        )
        new_obj_prob[torch.arange(new_xyz.shape[0]), object_id] = 1

        # make it a parameter
        # TODO: set requires grad to True
        new_obj_prob = nn.Parameter(new_obj_prob.requires_grad_(False))

        new_kf_id = torch.ones((new_xyz.shape[0])).int() * kf_idx  # [N,]
        new_nr_obs = torch.zeros((new_xyz.shape[0])).int()  # [N,]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            # new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_obj_prob=new_obj_prob,
            new_kf_idxs=new_kf_id,
            new_nr_obs=new_nr_obs,
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            # {
            #     "params": [self._features_rest],
            #     "lr": training_args.feature_lr / 20.0,
            #     "name": "f_rest",
            # },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        # for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
        #     l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        # f_rest = (
        #     self._features_rest.detach()
        #     .transpose(1, 2)
        #     .flatten(start_dim=1)
        #     .contiguous()
        #     .cpu()
        #     .numpy()
        # )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # TODO: store obj prob
        obj_prob = self._obj_prob.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation), axis=1
        )
        # attributes = np.concatenate(
        #     (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        # )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # extra_f_names = [
        #     p.name
        #     for p in plydata.elements[0].properties
        #     if p.name.startswith("f_rest_")
        # ]
        # extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape(
        #     (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        # )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float32, device=self.device).requires_grad_(
                True
            )
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float32, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        # self._features_rest = nn.Parameter(
        #     torch.tensor(features_extra, dtype=torch.float32, device=self.device)
        #     .transpose(1, 2)
        #     .contiguous()
        #     .requires_grad_(True)
        # )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacities, dtype=torch.float32, device=self.device
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                scales, dtype=torch.float32, device=self.device
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float32, device=self.device).requires_grad_(
                True
            )
        )

        # self.active_sh_degree = self.max_sh_degree
        self.max_radii_2d = torch.zeros((self._xyz.shape[0]), device=self.device)

        # cpu stored tensors
        self.kf_idx = torch.zeros((self._xyz.shape[0]), device="cpu").int()
        self.nr_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        #
        self._obj_prob = self._obj_prob[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii_2d = self.max_radii_2d[valid_points_mask]
        self.kf_idx = self.kf_idx[valid_points_mask.cpu()]
        self.nr_obs = self.nr_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        # new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_obj_prob,
        new_kf_idxs=None,
        new_nr_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            # "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        #
        self._obj_prob = torch.cat((self._obj_prob, new_obj_prob))

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii_2d = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        if new_kf_idxs is not None:
            self.kf_idx = torch.cat((self.kf_idx, new_kf_idxs)).int()

        if new_nr_obs is not None:
            self.nr_obs = torch.cat((self.nr_obs, new_nr_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        #
        new_obj_prob = self._obj_prob[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.kf_idx[selected_pts_mask.cpu()].repeat(N)
        new_nr_obs = self.nr_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            # new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_obj_prob=new_obj_prob,
            new_kf_idxs=new_kf_id,
            new_nr_obs=new_nr_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        #
        new_obj_prob = self._obj_prob[selected_pts_mask]

        new_kf_id = self.kf_idx[selected_pts_mask.cpu()]
        new_nr_obs = self.nr_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            # new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_obj_prob=new_obj_prob,
            new_kf_idxs=new_kf_id,
            new_nr_obs=new_nr_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii_2d > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
