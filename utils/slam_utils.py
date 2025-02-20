import torch

MASK_RGB_LOSS = True


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(
    render_image,
    render_depth,
    render_opacity,
    viewpoint,
    invert_depth=False,
    lambda_depth=0.9,
):
    gt_rgb = viewpoint.rgb  # 3xHxW
    gt_mask = viewpoint.mask if MASK_RGB_LOSS else torch.ones_like(gt_rgb[0])  # HxW
    gt_depth = viewpoint.depth[None]  # 1xHxW

    # Weight by opacity
    opacity_mask = render_opacity > 0.99

    # RGB computation
    rgb = torch.exp(viewpoint.exposure_a) * render_image + viewpoint.exposure_b
    rgb_mask = gt_mask * viewpoint.grad_mask * opacity_mask

    l1_rgb = render_opacity * torch.abs(rgb * rgb_mask - gt_rgb * rgb_mask).mean()
    l1_rgb = l1_rgb.mean()  # Ensure it's a scalar

    # Depth computation
    depth_mask = (gt_depth > 0) * opacity_mask

    if depth_mask.any():  # Only compute if depth_mask is not empty
        if invert_depth:
            eps = 1e-6  # To prevent division by zero
            l1_depth = torch.abs(
                (1 / (render_depth[depth_mask] + eps))
                - (1 / (gt_depth[depth_mask] + eps))
            ).mean()
        else:
            l1_depth = torch.abs(render_depth[depth_mask] - gt_depth[depth_mask]).mean()
    else:
        l1_depth = torch.tensor(
            0.0, device=render_depth.device, dtype=render_depth.dtype
        )  # Ensure correct device & dtype

    # return (lambda_depth * l1_rgb + (1 - lambda_depth) * l1_depth).mean()  # Explicitly ensure scalar
    return 0.5 * l1_rgb + l1_depth


def get_loss_mapping(
    render_image,
    render_depth,
    # render_opacity,
    viewpoint,
    init=False,
    invert_depth=False,
    lambda_depth=0.9,
):

    gt_rgb = viewpoint.rgb  # 3xHxW
    gt_mask = viewpoint.mask  # HxW
    gt_depth = viewpoint.depth[None]  # HxW
    if not MASK_RGB_LOSS:
        gt_mask = None

    # RGB

    if init:
        rgb = render_image
    else:
        rgb = (torch.exp(viewpoint.exposure_a)) * render_image + viewpoint.exposure_b

    rgb = rgb.permute(1, 2, 0)  # HxWx3
    gt_rgb = gt_rgb.permute(1, 2, 0)  # HxWx3

    if gt_mask is not None:
        l1_rgb = torch.abs(rgb[gt_mask] - gt_rgb[gt_mask])
    else:
        l1_rgb = torch.abs(rgb - gt_rgb)
    l1_rgb = l1_rgb.mean()

    # Depth

    # only use valid depth pixels
    depth_mask = gt_depth > 0

    if invert_depth:
        l1_depth = torch.abs(
            (1 / render_depth[depth_mask]) - (1 / gt_depth[depth_mask])
        )
    else:
        l1_depth = torch.abs(render_depth[depth_mask] - gt_depth[depth_mask])
    l1_depth = l1_depth.mean()

    return lambda_depth * l1_rgb + (1 - lambda_depth) * l1_depth


@torch.no_grad()
def get_median_depth(depth, mask=None, return_std=False):
    valid = depth > 0
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
