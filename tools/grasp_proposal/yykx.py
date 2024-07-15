import trimesh
import numpy as np
from .base import GraspProposal


class YYKX(GraspProposal):
    USD_PATH = "./mesh/yingyangkuaixian_converted/mesh_obj.usd"
    OBJ_PATH = "./mesh/yingyangkuaixian/mesh.obj"

    def __init__(self):
        mesh: trimesh.Trimesh = trimesh.load(self.OBJ_PATH, force="mesh")
        points = mesh.vertices
        vmin = points.min(axis=0)
        vmax = points.max(axis=0)
        super().__init__(vmin, vmax)
    
    def best_grasp(self, obj_wcT: np.ndarray, flip=False):
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 36, endpoint=False):
            grasp_pose = obj_wcT @ self.horizontal_proposal(theta)
            grasp_poses.append(grasp_pose)
        grasp_poses = np.stack(grasp_poses, axis=0)
        zaxes = grasp_poses[:, :3, 2]  # (N, 3)
        index = np.argmin(zaxes[:, -1])  # most downwards
        grasp_pose = grasp_poses[index]
        if flip:
            grasp_pose[:3, :2] *= -1
        return grasp_pose

