import torch
import numpy as np
from typing import Optional, Union
from ..perception import PinholeCamera


Array = Union[np.ndarray, torch.Tensor]


def ibvs(
    fp_cur: Array,
    Z_cur: Array,
    fp_tar: Array,
    Z_tar: Array,
    intrinsic: Optional[PinholeCamera] = None
) -> Array:
    """Image-based visual servoing

    Arguments:
    - fp_cur: (N, 2), 2 represents (x, y), features points of current camera frame; 
        * if intrinsic is None, assume fp_cur locates on current normalized camera plane
        * otherwise, assume fp_cur locates on pixel plane, will normalize fp_cur with intrinsic
    - Z_cur: (N,), depth of feature points in current camera frame
    - fp_tar: (N, 2), feature points of target camera frame
    - Z_tar: (N,), depth of feature points in target camera frame

    Returns:
    - vel: (6,), [v, w], camera velocity in current camera frame
    """

    assert fp_cur.shape == fp_tar.shape, "number of feature points not match"
    if intrinsic is not None:
        fp_cur = intrinsic.pixel_to_norm_camera_plane(fp_cur)
        fp_tar = intrinsic.pixel_to_norm_camera_plane(fp_tar)
    num_fp = fp_cur.shape[0]

    use_numpy = isinstance(fp_cur, np.ndarray)

    x_cur = fp_cur[:, 0]
    y_cur = fp_cur[:, 1]
    # build interaction matrix at current camera frame
    L_cur = (np.zeros((num_fp * 2, 6), fp_cur.dtype) if use_numpy else 
             torch.zeros(num_fp * 2, 6).to(fp_cur))
    L_cur[0::2, 0] = -1. / Z_cur
    L_cur[0::2, 2] = x_cur / Z_cur
    L_cur[0::2, 3] = x_cur * y_cur
    L_cur[0::2, 4] = -(1 + x_cur * x_cur)
    L_cur[0::2, 5] = y_cur
    L_cur[1::2, 1] = -1. / Z_cur
    L_cur[1::2, 2] = y_cur / Z_cur
    L_cur[1::2, 3] = 1 + y_cur * y_cur
    L_cur[1::2, 4] = -x_cur * y_cur
    L_cur[1::2, 5] = -x_cur

    x_tar = fp_tar[:, 0]
    y_tar = fp_tar[:, 1]
    # build interaction matrix at target camera frame
    L_tar = (np.zeros((num_fp * 2, 6), fp_tar.dtype) if use_numpy else 
             torch.zeros(num_fp * 2, 6).to(fp_tar))
    L_tar[0::2, 0] = -1. / Z_tar
    L_tar[0::2, 2] = x_tar / Z_tar
    L_tar[0::2, 3] = x_tar * y_tar
    L_tar[0::2, 4] = -(1 + x_tar * x_tar)
    L_tar[0::2, 5] = y_tar
    L_tar[1::2, 1] = -1. / Z_tar
    L_tar[1::2, 2] = y_tar / Z_tar
    L_tar[1::2, 3] = 1 + y_tar * y_tar
    L_tar[1::2, 4] = -x_tar * y_tar
    L_tar[1::2, 5] = -x_tar

    error = (np.zeros(num_fp * 2, fp_cur.dtype) if use_numpy else 
             torch.zeros(num_fp * 2).to(fp_cur))
    error[0::2] = x_tar - x_cur
    error[1::2] = y_tar - y_cur

    L_mean = (L_cur + L_tar) / 2.
    if use_numpy:
        vel: np.ndarray = np.linalg.lstsq(L_mean, error)[0]
    else:
        vel: torch.Tensor = torch.linalg.lstsq(L_mean, error).solution
    return vel
