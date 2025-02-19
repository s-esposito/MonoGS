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
        cur_viewpoint=None,
        cur_frame_idx=None,
        # gt_rgb=None,
        # gt_depth=None,
        # gt_segments=None,
        # gtnormal=None,
        viewpoints=None,
        keyframes=None,
        finish=False,
        cur_kf_list=None,
        unpack_buffers=False,
    ):
        # gaussians
        self.gaussians_dict = None
        self.kf_idx = None
        self.nr_obs = None

        if gaussians is not None:
            # activated values
            self.gaussians_dict = {}
            # get dynamic values
            with torch.no_grad():
                self.gaussians_dict["means"] = gaussians.get_xyz.detach().clone()  # (N, 3)
                self.gaussians_dict["rotations"] = gaussians.get_rotation.detach().clone()  # (N, 4)
                self.gaussians_dict["scales"] = gaussians.get_scaling.detach().clone()  # (N, 3)
                self.gaussians_dict["opacity"] = gaussians.get_opacity.detach().clone()  # (N, 1)
                self.gaussians_dict["features"] = gaussians.get_features.detach().clone()  # (N, C)
                obj_prob = gaussians.get_obj_prob.detach()
                obj_idx = torch.argmax(obj_prob, dim=1).clone()
                self.gaussians_dict["obj_idx"] = obj_idx  # (N)
            # get static values
            self.gaussians_dict["active_sh_degree"] = gaussians.active_sh_degree
            self.gaussians_dict["max_sh_degree"] = gaussians.max_sh_degree
            self.gaussians_dict["kf_idx"] = gaussians.kf_idx.clone()
            self.gaussians_dict["nr_obs"] = gaussians.nr_obs.clone()
            
            # for key in self.gaussians_dict.keys():
            #     if isinstance(self.gaussians_dict[key], torch.Tensor):
            #         print(f"key: {key}, shape: {self.gaussians_dict[key].shape}")

        # intrinisics
        self.cam_intrinsics = cam_intrinsics  # CameraIntrinsics
        self.cur_viewpoint = cur_viewpoint  # CameraExtrinsics
        self.cur_frame_idx = cur_frame_idx  # int
        self.viewpoints = viewpoints  # dict of CameraExtrinsics (all frames)
        self.keyframes = keyframes  # list of CameraExtrinsics (window)
        
        if cur_viewpoint is not None and unpack_buffers:
            # get rgb
            if cur_viewpoint.rgb is not None:
                gt_rgb = cur_viewpoint.rgb
            else:
                raise ValueError("RGB must be provided")
            
            # get depth
            if cur_viewpoint.depth is not None:
                gt_depth = cur_viewpoint.depth
            else:
                raise ValueError("Depth must be provided")
            
            # # get mask
            # if cur_viewpoint.mask is not None:
            #     gt_mask = cur_viewpoint.mask
            # else:
            #     gt_mask = torch.ones_like(gt_rgb[0]).bool()

            # get segmentation
            if cur_viewpoint.segmentation is not None:
                gt_segments = cur_viewpoint.segmentation
            else:
                gt_segments = torch.zeros_like(gt_rgb[0]).long()
                
            self.gt_rgb = self.resize_img(gt_rgb, 212, bilinear=True)
            self.gt_depth = self.resize_img(gt_depth, 212, bilinear=True)
            self.gt_segments = self.resize_img(gt_segments, 212, bilinear=False)
        else:
            self.gt_rgb = None
            self.gt_depth = None
            self.gt_segments = None
        # self.gtnormal = self.resize_img(gtnormal, 320)

        self.finish = finish
        self.cur_kf_list = cur_kf_list  # list of int (viewpoints indices)

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

    # def get_covariance(self, scaling_modifier=1):
    #     return self.build_covariance_from_scaling_rotation(
    #         self.get_scaling, scaling_modifier, self._rotation
    #     )

    # def build_covariance_from_scaling_rotation(
    #     self, scaling, scaling_modifier, rotation
    # ):
    #     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    #     actual_covariance = L @ L.transpose(1, 2)
    #     symm = strip_symmetric(actual_covariance)
    #     return symm
