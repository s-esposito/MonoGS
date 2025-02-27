#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.camera_utils import (
    CameraExtrinsics,
    CameraIntrinsics,
    get_full_proj_transform,
)
# from gaussian_splatting.scene.gaussian_model import GaussianModel
# from gaussian_splatting.utils.sh_utils import eval_sh


def render(
    viewpoint_camera: CameraExtrinsics,
    cam_intrinsics: CameraIntrinsics,
    # gaussians: GaussianModel,
    means: torch.Tensor,  # xyz
    rotations: torch.Tensor,  # rot
    scales: torch.Tensor,  # scale
    opacity: torch.Tensor,  # opacity
    features: torch.Tensor,  # rgb or sh
    # active_sh_degree: int,
    # pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if means.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device="cuda")
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(cam_intrinsics.FoVx * 0.5)
    tanfovy = math.tan(cam_intrinsics.FoVy * 0.5)

    full_proj_transform = get_full_proj_transform(
        cam_extrinsics=viewpoint_camera, cam_intrinsics=cam_intrinsics
    )
    projection_matrix = cam_intrinsics.projection_matrix

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam_intrinsics.height),
        image_width=int(cam_intrinsics.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_transform,
        projmatrix_raw=projection_matrix,
        sh_degree=0,  # active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    means3D = means
    means2D = screenspace_points
    # opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None

    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    # else:

    # check if the covariance is isotropic
    if scales.shape[-1] == 1:
        scales = scales.repeat(1, 3)
    else:
        scales = scales
    # rotations = gaussians.get_rotation

    # If precomputed colors are provided, use them. 
    # Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # colors_precomp = None
    # if colors_precomp is None:
    #     # if pipe.convert_SHs_python:
    #     #     shs_view = features.transpose(1, 2).view(
    #     #         -1, 3, (gaussians.max_sh_degree + 1) ** 2
    #     #     )
    #     #     dir_pp = gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
    #     #         features.shape[0], 1
    #     #     )
    #     #     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #     #     sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
    #     #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     # else:
    #     shs = features
    # else:
    #     colors_precomp = override_color
    # shs = features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if mask is not None:
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=None,  # shs[mask],
            colors_precomp=features[mask],  # colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=None,  # cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,  # sh,
            colors_precomp=features,  # colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,  # cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }
