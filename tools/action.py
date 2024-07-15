import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(x):
    return np.array([[0, -x[2], x[1]], 
                     [x[2], 0, -x[0]], 
                     [-x[1], x[0], 0]])


def sinx_div_x(x):
    return np.sinc(x / np.pi)


def one_minus_cosx_div_x(x: float) -> float:
    """y = (1 - cos(x)) / x; Use Taylor expansion when near zero."""
    if np.issubdtype(type(x), np.float32):
        eps = 1e-2
    else:
        eps = 1e-7

    if np.abs(x) < eps:
        x2 = x*x; x3 = x*x2; x5 = x3*x2
        y = x/2 - x3/24 + x5/720
    else:
        y = (1 - np.cos(x)) / x
    return y


def left_jacobian_of_R(u: np.ndarray) -> np.ndarray:
    phi = np.linalg.norm(u)
    a = u / (phi + 1e-16); a_ = a.reshape(3, 1)
    # https://zhuanlan.zhihu.com/p/497849222
    J = sinx_div_x(phi) * np.eye(3, dtype=u.dtype) + \
        (1 - sinx_div_x(phi)) * a_ @ a_.T + \
        one_minus_cosx_div_x(phi) * skew(a)
    return J


def velocity_integrate(vel: np.ndarray, dt: float):
    v, w = vel[:3], vel[3:]
    J = left_jacobian_of_R(w * dt)
    
    dT = np.eye(4, dtype=vel.dtype)
    dT[:3, :3] = R.from_rotvec(w * dt).as_matrix()
    dT[:3, 3] = J @ (v * dt)
    return dT


def action(wcT: np.ndarray, vel: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
    Arguments:
    - wcT: camera extrinsic, ^{world}_{cam} T, 4x4 transformation matrix
    - vel: [v, w], shape = (6,), ^{cam} vel _{cam}, camera velocity in current camera frame
    - dt: integration time

    Returns:
    - wcT: new camera extrinsic
    """
    wcT = wcT @ velocity_integrate(vel, dt)
    wcT[:3, :3] = R.from_matrix(wcT[:3, :3]).as_matrix()
    return wcT


def find_action(ctT: np.ndarray, dt: float = 1.0):
    """
    Argumets:
    - ctT: target camera pose in current frame, ^{current}_{target} T, 4x4 matrix
    - dt: integration time

    Returns:
    - vel: [v, w], shape = (6,), ^{cam} vel _{cam}, camera velocity in current camera frame
    """
    cwc_mul_dt = R.from_matrix(ctT[:3, :3]).as_rotvec()
    J = left_jacobian_of_R(cwc_mul_dt)
    cvc_mul_dt = np.linalg.inv(J) @ ctT[:3, 3]
    return np.concatenate([cvc_mul_dt, cwc_mul_dt]) / dt
