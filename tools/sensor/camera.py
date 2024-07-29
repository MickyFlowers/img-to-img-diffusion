import numpy as np

class Camera:
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def backproject(self, frame: np.ndarray) -> np.ndarray:
        depth_img = frame
        h, w = depth_img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        x = ((u - self.cx) * depth_img / self.fx).astype(np.float32)
        y = ((v - self.cy) * depth_img / self.fy).astype(np.float32)
        z = depth_img.astype(np.float32)

        return np.stack([x, y, z], axis=-1)

    def projet(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        u = (x * self.fx / z + self.cx).astype(np.int)
        v = (y * self.fy / z + self.cy).astype(np.int)
        return np.stack([u, v], axis=-1)

    def pixel_to_camera_frame(self, pixel: np.ndarray):
        u, v = pixel[:, 0], pixel[:, 1]
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return np.stack([x, y], axis=-1)
