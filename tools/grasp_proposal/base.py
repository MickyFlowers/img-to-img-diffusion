
import numpy as np


class GraspProposal(object):
    def __init__(self, vmin: np.ndarray, vmax: np.ndarray):
        self.vmin = vmin
        self.vmax = vmax

        self.center3d = (self.vmin + self.vmax) / 2.0
        self.center2d = np.array([self.center3d[0], self.center3d[1], 0])
        self.radius = np.linalg.norm((vmax - vmin)[:2]) / (2 * np.sqrt(2))
    
    def top_down_border_proposal(self, theta):
        pos = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            self.vmax[-1]
        ]) + self.center2d

        axis_x = np.array([np.cos(theta), np.sin(theta), 0])
        axis_y = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2), 0])
        axis_z = np.array([0, 0, -1])
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
    
    def top_down_center_proposal(self, theta):
        pos = np.array([0, 0, self.vmax[-1]]) + self.center2d
        axis_x = np.array([np.cos(theta), np.sin(theta), 0])
        axis_y = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2), 0])
        axis_z = np.array([0, 0, -1])
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose

    def horizontal_proposal(self, theta, h_ratio=0.5):
        pos = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            self.vmin[-1] + h_ratio * (self.vmax[-1] - self.vmin[-1])
        ]) + self.center2d
        pos2d = np.array([pos[0], pos[1], 0])

        axis_y = np.array([0, 0, -1])
        axis_z = self.center2d - pos2d
        axis_z = axis_z / np.linalg.norm(axis_z, axis=-1, keepdims=True)
        axis_x = np.cross(axis_y, axis_z)
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
    
    def best_grasp(self, obj_wcT: np.ndarray, condition = None):
        raise NotImplementedError
