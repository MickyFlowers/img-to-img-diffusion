import numpy as np
from scipy.spatial.transform import Rotation as R

def samplePose(pos_upper, pos_lower, ori_upper, ori_lower):
    pos = np.random.uniform(pos_lower, pos_upper)
    ori = np.random.uniform(ori_lower, ori_upper)
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', ori).as_matrix()
    T[:3, 3] = pos
    return pos, ori, T