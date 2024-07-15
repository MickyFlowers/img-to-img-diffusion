import torch
import numpy as np
from torch import Tensor
from einops import rearrange
from typing import Optional, Union
from torchvision.transforms import v2
from ..perception import PinholeCamera


def grad_image(image: Tensor):
    # image: (B, C, H, W)
    gy, gx = torch.gradient(image, dim=(-2, -1))
    grad: Tensor = torch.stack([gx, gy], axis=-1)  # (B, C, H, W, 2)
    return grad


def compute_LI(gI: Tensor, fp_norm: Tensor, Z: Tensor):
    """
    Arguments:
    - gI: (B, C, H, W, 2)
    - fp_norm: (2, H, W) coordinates on normalized camera plane
    - Z: Tensor of shape (B,) or (B, 1, H, W)
    - intrinsic: PinholeCamera

    Returns:
    - L: (B, C, H, W, 6)
    """
    x, y = fp_norm.unbind(0)
    dx = (x[0, 1:] - x[0, :-1]).mean()
    dy = (y[1:, 0] - y[:-1, 0]).mean()

    Ix = gI[..., 0] / dx  # (B, C, H, W)
    Iy = gI[..., 1] / dy  # (B, C, H, W)

    if Z.dim() == 1:
        assert Z.shape[0] == Ix.shape[0]  # batch_size
        Z = Z[:, None, None, None]  # (B, 1, 1, 1)
    else:
        assert Z.shape[2:] == Ix.shape[2:]  # H, W

    L = torch.stack(
        [
            Ix / Z, 
            Iy / Z, 
            -(x * Ix + y * Iy) / Z, 
            -Ix * x * y - (1 + y * y) * Iy, 
            (1 + x * x) * Ix + Iy * x * y, 
            Iy * x - Ix * y
        ],
        dim=-1
    )  # L = -gradI^T @ Lx
    return L


def jac_Iv(image: Tensor, fp_norm: Tensor, Z: Tensor):
    gI = grad_image(image)  # (B, C, H, W, 2)
    LI = compute_LI(gI, fp_norm, Z)  # (B, C, H, W, 6)
    return LI


def lm_step(J: Tensor, r: Tensor, damp: float):
    """
    solve x for Jx = r
    
    - J: (C*H*W, 6)
    - r: (C*H*W,)
    """
    H = J.T @ J
    D = torch.diag(torch.diag(H))
    A = H + damp * D
    b = J.T @ r
    x: Tensor = torch.linalg.solve(A, b)
    return x


def pvs_impl(
    I_cur: Tensor, 
    I_tar: Tensor, 
    fp_norm: Tensor, 
    Z: Tensor, 
    damp: Union[float, Tensor], 
    border: int = 10
):
    """
    Arguments: 
    - I_cur, I_tar: (B, C, H, W), color image or feature map
    - fp_norm: (2, H, W), coordinates on normalized camera plane, [x, y]
    - Z: Tensor of shape (B,) or (B, 1, H, W), depth image
    - damp: float or Tensor of shape (B,)
    - border: number of pixels to discard of image gradients

    Returns:
    - vel: (B, 6), [v, w], camera velocity in camera frame
    - cost: (B,), sum of residuals in LM steps, can be an indicator to adjust `damp`
    """
    B, C, H, W = I_cur.shape
    LI = jac_Iv(I_cur, fp_norm, Z)
    # LI = jac_Iv(I_tar, Z, intrinsic)
    # LI = 0.5 * (jac_Iv(I_cur, Z, intrinsic) + jac_Iv(I_tar, Z, intrinsic))
    err = I_cur - I_tar

    assert (H - border*2) > 3
    assert (W - border*2) > 3

    LI = LI[:, :, border:H-border, border:W-border].contiguous().view(B, -1, 6)
    err = err[:, :, border:H-border, border:W-border].contiguous().view(B, -1)

    if not isinstance(damp, torch.Tensor):
        damp = torch.tensor([damp]).to(I_cur).expand(B)
    
    if torch.__version__.startswith("2."):
        vel = -torch.vmap(lm_step)(LI, err, damp)  # (B, 6)
    else:
        vel = -torch.stack([lm_step(J, r, d) for J, r, d in zip(LI, err, damp)])
    cost = 0.5 * (err ** 2).sum(dim=-1)  # (B,)
    return vel, cost


def pvs_batch_torch(
    I_cur: Tensor,
    I_tar: Tensor,
    Z: Tensor,
    intrinsic: PinholeCamera, 
    damp: Union[float, Tensor], 
    border: int = 10,
    color_transform: Optional[v2.Transform] = None,
    shape_transform: Optional[v2.Transform] = None
):
    """
    Arguments: 
    - I_cur, I_tar: (B, C, H, W), color image or feature map
    - Z: Tensor of shape (B,) or (B, 1, H, W), depth image
    - intrinsic: PinholeCamera instance
    - damp: float or Tensor of shape (B,)
    - border: number of pixels to discard of image gradients
    - color_transform: torchvision Transforms applied to image
    - shape_transform: torchvision Transforms applied to image and depth

    Returns:
    - vel: (B, 6), [v, w], camera velocity in camera frame
    - cost: (B,), sum of residuals in LM steps, can be an indicator to adjust `damp`
    """
    B, C, H, W = I_cur.shape
    assert (H, W) == (intrinsic.height, intrinsic.width), (
        "image shape = {}, intrinsic = {}".format(I_cur.shape, intrinsic))

    xv, yv = torch.meshgrid(
        torch.arange(intrinsic.width).to(I_cur), 
        torch.arange(intrinsic.height).to(I_tar), 
        indexing="xy"
    )
    fp: Tensor = torch.stack([xv, yv], dim=-1)  # (H, W, 2)
    fp_norm = intrinsic.pixel_to_norm_camera_plane(fp)
    fp_norm = rearrange(fp_norm, "h w c -> c h w")

    if color_transform is not None:
        I_cur: Tensor = color_transform(I_cur)
        I_tar: Tensor = color_transform(I_tar)
    
    if shape_transform is not None:
        I_cur: Tensor = shape_transform(I_cur)
        I_tar: Tensor = shape_transform(I_tar)
        fp_norm: Tensor = shape_transform(fp_norm[None])[0]  # (2, H, W)

        if Z.dim() == 4:
            Z = shape_transform(Z)
    
    return pvs_impl(
        I_cur=I_cur,
        I_tar=I_tar,
        fp_norm=fp_norm,
        Z=Z,
        damp=damp,
        border=border
    )


def pvs_single_numpy(
    I_cur: np.ndarray,
    I_tar: np.ndarray,
    Z: Union[float, np.ndarray],
    intrinsic: PinholeCamera, 
    damp: float, 
    border: int = 10,
    color_transform: Optional[v2.Transform] = None,
    shape_transform: Optional[v2.Transform] = None,
    device="cpu"
):
    """
    Arguments: 
    - I_cur, I_tar: (H, W, C), color image
    - Z: float or Tensor of shape (H, W), depth image
    - intrinsic: PinholeCamera instance
    - damp: float
    - border: number of pixels to discard of image gradients
    - color_transform: torchvision Transforms applied to image
    - shape_transform: torchvision Transforms applied to image and depth
    - device: str or torch.device

    Returns:
    - vel: (6), [v, w], camera velocity in camera frame
    - cost: float, sum of residuals in LM steps, can be an indicator to adjust `damp`
    """
    div_cur = 255.0 if (I_cur.dtype == np.uint8) else 1.0
    div_tar = 255.0 if (I_tar.dtype == np.uint8) else 1.0
    I_cur = torch.from_numpy(I_cur).float().to(device) / div_cur
    I_tar = torch.from_numpy(I_tar).float().to(device) / div_tar
    I_cur = rearrange(I_cur, "h w c -> 1 c h w")
    I_tar = rearrange(I_tar, "h w c -> 1 c h w")

    if isinstance(Z, np.ndarray):
        Z = torch.from_numpy(Z).float().to(device)
        Z = rearrange(Z, "h w -> 1 1 h w")
    else:
        Z = torch.tensor([Z]).float().to(device)  # (1,)

    with torch.no_grad():
        vel, cost = pvs_batch_torch(
            I_cur=I_cur,
            I_tar=I_tar,
            Z=Z,
            intrinsic=intrinsic,
            damp=damp,
            border=border,
            color_transform=color_transform,
            shape_transform=shape_transform
        )
    
    vel: np.ndarray = vel[0].detach().cpu().numpy()
    cost = cost[0].item()
    return vel, cost

