import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_matrix(q: np.ndarray):
    return Rotation.from_quat(q).as_matrix()


def matrix_to_quaternion(mat: np.ndarray):
    return Rotation.from_matrix(mat).as_quat()


def euler_angles_to_matrix(a: np.ndarray, convention: str):
    return Rotation.from_euler(convention, a, degrees=False).as_matrix()


def matrix_to_euler_angles(mat: np.ndarray, convention: str):
    return Rotation.from_matrix(mat).as_euler(convention, degrees=False)


def _copysign(a: np.ndarray, b: np.ndarray):
    signs_differ = (a < 0) != (b < 0)
    return np.where(signs_differ, -a, a)


def random_quaternions(n: int):
    o = np.random.randn(n, 4)
    s = (o * o).sum(1)
    o = o / _copysign(np.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(n: int):
    return quaternion_to_matrix(random_quaternions(n))


def random_rotation():
    return random_quaternions(1)[0]


def axis_angle_to_matrix(a: np.ndarray):
    return Rotation.from_rotvec(a).as_matrix()


def matrix_to_axis_angle(mat: np.ndarray):
    return Rotation.from_matrix(mat).as_rotvec()


def axis_angle_to_quaternion(a: np.ndarray):
    return Rotation.from_rotvec(a).as_quat()


def quaternion_to_axis_angle(q: np.ndarray):
    return Rotation.from_rotvec(q).as_rotvec()


def _normalize(x: np.ndarray, dim=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=dim) + eps)


def rotation_6d_to_matrix(d6: np.ndarray):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1, dim=-1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2, dim=-1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(mat: np.ndarray):
    batch_dim = mat.shape()[:-2]
    return mat[..., :2, :].reshape(batch_dim + (6,))

