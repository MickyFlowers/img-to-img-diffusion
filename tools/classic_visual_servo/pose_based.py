import torch
import numpy as np
from typing import Union
from ..rot_transforms import numpy as rot_numpy
from ..rot_transforms import torch as rot_torch


Array = Union[np.ndarray, torch.Tensor]


def get_backend(x: Array):
    if isinstance(x, np.ndarray):
        return np, rot_numpy, True
    elif isinstance(x, torch.Tensor):
        return torch, rot_torch, False
    else:
        raise TypeError("x should be np.ndarray or torch.Tensor, got {}".format(type(x)))


def pbvs_center(cur_wcT: Array, tar_wcT: Array, wPo: Array):
    """PBVS1: ensure the projection of scene center is always at center of camera's FoV

    Arguments:
    - cur_wcT: current camera pose in world frame, 4x4 transformation matrix
    - tar_wcT: target camera pose in world frame, 4x4 trasformation matrix
    - wPo: scene center position in world frame, shape (3,)

    Returns:
    - vel: [v, w], camera velocity in camera frame
    """
    backend, R, use_numpy = get_backend(cur_wcT)

    # w: world frame, o: center, P: points
    tar_cwT = backend.linalg.inv(tar_wcT)
    tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
    # t: target camera frame, c: current camera frame, w: world frame

    cur_cwT = backend.linalg.inv(cur_wcT)
    cPo = wPo @ cur_cwT[:3, :3].T + cur_cwT[:3, 3]

    tcT = tar_cwT @ cur_wcT
    u = R.matrix_to_axis_angle(tcT[:3, :3])

    kwargs = {"axis": -1} if use_numpy else {"dim": -1}
    v = -(tPo - cPo + backend.cross(cPo, u, **kwargs))
    w = -u
    vel = backend.concatenate([v, w])
    return vel


def pbvs_straight(cur_wcT: Array, tar_wcT: Array):
    """PBVS2: goes straight and shortest path

    Arguments:
    - cur_wcT: current camera pose in world frame, 4x4 transformation matrix
    - tar_wcT: target camera pose in world frame, 4x4 trasformation matrix

    Returns:
    - vel: [v, w], camera velocity in camera frame
    """
    backend, R, _ = get_backend(cur_wcT)
    tcT = backend.linalg.inv(tar_wcT) @ cur_wcT
    u = R.matrix_to_axis_angle(tcT[:3, :3])

    v = -tcT[:3, :3].T @ tcT[:3, 3]
    w = -u
    vel = backend.concatenate([v, w])
    return vel


def skew(x: Array):
    if isinstance(x, np.ndarray):
        return np.array([[0, -x[2], x[1]], 
                         [x[2], 0, -x[0]], 
                         [-x[1], x[0], 0]]).astype(x.dtype)
    else:
        return torch.tensor([[0, -x[2], x[1]], 
                             [x[2], 0, -x[0]], 
                             [-x[1], x[0], 0]]).to(x)


def inv(J: Array, damp=1e-3, weight: Array = None):
    backend, _, _ = get_backend(J)

    if weight is None:
        Jt = J.T
    else:
        if weight.ndim == 1:
            weight = backend.diag(weight)
        Jt = J.T @ weight

    JtJ = Jt @ J
    D = backend.diag(backend.diag(JtJ))
    A = JtJ + damp * D
    return backend.linalg.inv(A) @ Jt


def pbvs_hybrid(cur_wcT: Array, tar_wcT: Array, wPo: Array):
    """
    Ref: 2 1/2 D visual servoing: a possible solution to improve 
    image-based and position-based visual servoing. ICRA 2000.

    Arguments:
    - cur_wcT: current camera pose in world frame, 4x4 transformation matrix
    - tar_wcT: target camera pose in world frame, 4x4 trasformation matrix
    - wPo: scene center position in world frame, shape (3,)

    Returns:
    - vel: [v, w], camera velocity in camera frame
    """

    backend, R, use_numpy = get_backend(cur_wcT)
    
    # w: world frame, o: center, P: points
    # t: target camera frame, c: current camera frame, w: world frame
    tar_cwT = backend.linalg.inv(tar_wcT)
    tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
    t_xg = tPo[:2] / tPo[2:3]  # scene center in target normalized camera plane

    cur_cwT = backend.linalg.inv(cur_wcT)
    cPo = wPo @ cur_cwT[:3, :3].T + cur_cwT[:3, 3]
    c_xg = cPo[:2] / cPo[2:3]  # scene center in current normalized camera plane

    tcT = tar_cwT @ cur_wcT
    tcR = tcT[:3, :3]
    u = R.matrix_to_axis_angle(tcR)

    e = backend.concatenate([tcT[:3, 3], c_xg - t_xg, u[2:3]])
    x, y = c_xg
    Z = cPo[2]

    theta = backend.linalg.norm(u)
    u_unit = u / (theta + 1e-7)
    Ux = skew(u_unit)
    I3 = np.eye(3, dtype=u.dtype) if use_numpy else torch.eye(3).to(u)
    Jw0 = I3 - theta/2 * Ux + (
        1 - backend.sinc(theta/backend.pi) / backend.sinc(theta/backend.pi/2)**2) * (Ux @ Ux)

    Jwv = [[-1, 0, x],
           [0, -1, y],
           [0, 0, 0]]
    Jw = [[x*y, -(1+x*x), y],
          [1+y*y, -x*y, -x],
          Jw0[2, :]]
    
    Jwv = np.array(Jwv, dtype=x.dtype) if use_numpy else torch.tensor(Jwv).to(x)
    Jw = np.array(Jw, dtype=x.dtype) if use_numpy else torch.tensor(Jw).to(x)
    
    # L = |       R   0 |   =>   L^* = |                R^T      0 |
    #     | 1/Z*Jwv  Jw |              | -1/Z * Jw^-1 * R^T  Jw^-1 |
    
    L_inv = np.zeros((6, 6), e.dtype) if use_numpy else torch.zeros(6, 6).to(e)
    Jw_inv = backend.linalg.inv(Jw)
    L_inv[:3, :3] = tcR.T
    L_inv[3:, :3] = -Jw_inv @ Jwv @ tcR.T / Z
    L_inv[3:, 3:] = Jw_inv

    vel = -L_inv @ e
    return vel

