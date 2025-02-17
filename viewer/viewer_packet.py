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
        gt_segments=None,
        # gtnormal=None,
        viewpoints=None,
        keyframes=None,
        finish=False,
        kf_window=None,
    ):
        # gaussians
        self.gaussians = None
        self.unique_kfIDs = None
        self.n_obs = None

        if gaussians is not None:
            # activated values
            self.gaussians = {}
            with torch.no_grad():
                self.gaussians["means"] = gaussians.get_xyz.detach().clone()
                self.gaussians["rotations"] = gaussians.get_rotation.detach().clone()
                self.gaussians["scales"] = gaussians.get_scaling.detach().clone()
                self.gaussians["opacity"] = gaussians.get_opacity.detach().clone()
                self.gaussians["features"] = gaussians.get_features.detach().clone()
                self.gaussians["active_sh_degree"] = gaussians.active_sh_degree
                self.gaussians["max_sh_degree"] = gaussians.max_sh_degree
                self.gaussians["ids"] = gaussians.get_instance_id.detach().clone()
                self.gaussians["unique_kfIDs"] = gaussians.unique_kfIDs.clone()
                self.gaussians["n_obs"] = gaussians.n_obs.clone()

            # self.has_gaussians = True
            # self.get_xyz = gaussians.get_xyz.detach().clone()
            # self.active_sh_degree = gaussians.active_sh_degree
            # self.get_opacity = gaussians.get_opacity.detach().clone()
            # self.get_scaling = gaussians.get_scaling.detach().clone()
            # self.get_rotation = gaussians.get_rotation.detach().clone()
            # self.max_sh_degree = gaussians.max_sh_degree
            # self.get_features = gaussians.get_features.detach().clone()
            # self.get_instance_id = gaussians.get_instance_id.detach().clone()

            # self._rotation = gaussians._rotation.detach().clone()
            # self.rotation_activation = torch.nn.functional.normalize
            # self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            # self.n_obs = gaussians.n_obs.clone()

        # intrinisics
        self.cam_intrinsics = cam_intrinsics  # CameraIntrinsics

        # self.gt_keyframes = gt_keyframes  # list of CameraExtrinsics
        self.current_frame = current_frame  # CameraExtrinsics
        self.current_frame_idx = current_frame_idx  # int
        self.viewpoints = viewpoints  # dict of CameraExtrinsics (all frames)
        self.keyframes = keyframes  # list of CameraExtrinsics (window)

        self.gt_rgb = self.resize_img(gt_rgb, 212, bilinear=True)
        self.gt_depth = self.resize_img(gt_depth, 212, bilinear=True)
        self.gt_segments = self.resize_img(gt_segments, 212, bilinear=False)
        # self.gtnormal = self.resize_img(gtnormal, 320)

        self.finish = finish
        self.kf_window = kf_window  # list of int (viewpoints indices)

    def resize_img(self, img, width, bilinear=True):

        if img is None:
            return None

        # # check if img is numpy
        # if isinstance(img, np.ndarray):
        #     height = int(width * img.shape[0] / img.shape[1])
        #     return cv2.resize(img, (width, height))

        if img.ndim == 2:
            img = img.unsqueeze(0)  # 1xHxW
        else:
            pass  # 3xHxW

        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        
        converted = False
        if img.dtype == torch.int or img.dtype == torch.int32 or img.dtype == torch.int64:
            img = img.float()
            converted = True
        
        if bilinear:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            )
        else:
            # do not interpolate
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(height, width), mode="nearest"
            )
            
        if converted:
            # convert back
            img = img.int()
        
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
