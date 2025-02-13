import torch
import cv2
import numpy as np
from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)


class MainToViewerPacket:
    def __init__(
        self,
        gaussians=None,
        cam_intrinsics=None,
        current_frame=None,
        current_frame_idx=None,
        gt_rgb=None,
        gt_depth=None,
        gtnormal=None,
        viewpoints=None,
        keyframes=None,
        finish=False,
        kf_window=None,
    ):
        # gaussians
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True
            self.get_xyz = gaussians.get_xyz.detach().clone()
            self.active_sh_degree = gaussians.active_sh_degree
            self.get_opacity = gaussians.get_opacity.detach().clone()
            self.get_scaling = gaussians.get_scaling.detach().clone()
            self.get_rotation = gaussians.get_rotation.detach().clone()
            self.max_sh_degree = gaussians.max_sh_degree
            self.get_features = gaussians.get_features.detach().clone()
            self.get_ids = gaussians.get_ids.detach().clone()

            self._rotation = gaussians._rotation.detach().clone()
            self.rotation_activation = torch.nn.functional.normalize
            self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            self.n_obs = gaussians.n_obs.clone()

        # intrinisics
        self.cam_intrinsics = cam_intrinsics  # CameraIntrinsics

        # self.gt_keyframes = gt_keyframes  # list of CameraExtrinsics
        self.current_frame = current_frame  # CameraExtrinsics
        self.current_frame_idx = current_frame_idx  # int
        self.viewpoints = viewpoints  # dict of CameraExtrinsics (all frames)
        self.keyframes = keyframes  # list of CameraExtrinsics (window)

        self.gt_rgb = self.resize_img(gt_rgb, 320)
        self.gt_depth = self.resize_img(gt_depth, 320)
        self.gtnormal = self.resize_img(gtnormal, 320)

        self.finish = finish
        self.kf_window = kf_window  # list of int (viewpoints indices)

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return img.squeeze(0)

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm