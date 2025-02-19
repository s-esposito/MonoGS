import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix, getWorld2View
from utils.slam_utils import image_gradient, image_gradient_mask


class CameraIntrinsics(nn.Module):
    def __init__(
        self,
        fx,
        fy,
        cx,
        cy,
        height,
        width,
        device="cuda:0",
    ):
        super(CameraIntrinsics, self).__init__()
        self.cx = cx
        self.cy = cy
        self.height = height
        self.width = width
        self.device = device

        # TODO: test with requires_grad=True
        self.fx = nn.Parameter(torch.tensor([fx], requires_grad=False, device=device))
        self.fy = nn.Parameter(torch.tensor([fy], requires_grad=False, device=device))

    @property
    def FoVx(self):
        return 2 * torch.atan(self.width / (2 * self.fx)).cpu().item()

    @property
    def FoVy(self):
        return 2 * torch.atan(self.height / (2 * self.fy)).cpu().item()

    @property
    def projection_matrix(self):
        return getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.width,
            H=self.height,
        ).transpose(0, 1)

    @staticmethod
    def init_from_dataset(dataset):
        return CameraIntrinsics(
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(
        fx,
        fy,
        cx,
        cy,
        height,
        width,
    ):
        return CameraIntrinsics(
            fx,
            fy,
            cx,
            cy,
            height,
            width,
        )


class CameraExtrinsics(nn.Module):
    def __init__(
        self,
        frame_idx: int,
        device: str,
        rgb=None,
        depth=None,
        mask=None,
        segmentation=None,
        gt_pose=None,
    ):
        super(CameraExtrinsics, self).__init__()
        self.frame_idx = frame_idx
        self.device = device

        T = torch.eye(4, device=device, dtype=torch.float32)
        self.R = T[:3, :3]
        self.T = T[:3, 3]

        if gt_pose is None:
            self.R_gt = None
            self.T_gt = None
        else:
            self.R_gt = gt_pose[:3, :3].to(dtype=torch.float32)
            self.T_gt = gt_pose[:3, 3].to(dtype=torch.float32)

        self.rgb = rgb
        self.mask = mask
        self.depth = depth
        self.segmentation = segmentation
        self.grad_mask = None

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

    @staticmethod
    def init_from_dataset(
        dataset,
        frame_idx,
    ):
        data = dataset[frame_idx]

        rgb = data.get("rgb")
        assert rgb is not None, "rgb is required"

        segmentation = data.get("segmentation", None)
        mask = data.get("mask", None)
        depth = data.get("depth", None)
        pose = data.get("pose", None)

        if mask is not None:
            rgb = rgb * mask
            depth = depth * mask

        viewpoint = CameraExtrinsics(
            frame_idx,
            rgb=rgb,
            depth=depth,
            mask=mask,
            segmentation=segmentation,
            gt_pose=pose,
            device=dataset.device,
        )

        return viewpoint

    @staticmethod
    def init_from_gui(
        frame_idx,
        pose,
    ):
        return CameraExtrinsics(
            frame_idx,
            rgb=None,
            gt_pose=pose,
            device="cuda:0",
        )

    @property
    def world_view_transform(self):
        return getWorld2View(self.R, self.T).transpose(0, 1)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = 1.1  # config["Training"]["edge_threshold"]

        gray_img = self.rgb.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        # if config["Dataset"]["type"] == "replica":
        #     row, col = 32, 32
        #     multiplier = edge_threshold
        #     _, h, w = self.rgb.shape
        #     for r in range(row):
        #         for c in range(col):
        #             block = img_grad_intensity[
        #                 :,
        #                 r * int(h / row) : (r + 1) * int(h / row),
        #                 c * int(w / col) : (c + 1) * int(w / col),
        #             ]
        #             th_median = block.median()
        #             block[block > (th_median * multiplier)] = 1
        #             block[block <= (th_median * multiplier)] = 0
        #     self.grad_mask = img_grad_intensity
        # else:
        
        median_img_grad_intensity = img_grad_intensity.median()
        self.grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )

    # def clean(self):
    #     self.rgb = None
    #     self.depth = None
    #     self.grad_mask = None
    #     self.cam_rot_delta = None
    #     self.cam_trans_delta = None
    #     self.exposure_a = None
    #     self.exposure_b = None


def get_full_proj_transform(
    cam_extrinsics: CameraExtrinsics, cam_intrinsics: CameraIntrinsics
):
    world_view_transform = cam_extrinsics.world_view_transform
    projection_matrix = cam_intrinsics.projection_matrix
    return (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
