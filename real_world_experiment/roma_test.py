import sys

sys.path.append("./")
from xlib.algo.kp_matcher import RomaMatchAlgo, KpMatchAlgo
import cv2
import numpy as np
import torch

img1 = "real_world_experiment/ref_img/00837.jpg"
img2 = "real_world_experiment/ref_img/Out_00837_resized.jpg"
img1 = cv2.imread(img1)
img2 = cv2.imread(img2)
mask = np.load(
    "/home/cyx/chenyuxi/project/img-to-img-diffusion/real_world_experiment/ref_img/mask.npy"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

kpmatcher = RomaMatchAlgo("tiny_roma_v1_outdoor", device=device)
# kpmatcher = KpMatchAlgo()
# from xlib.device.sensor.camera import RealSenseCamera
# camera = RealSenseCamera()
_, _, match_img = kpmatcher.match(img1, img2, mask)
cv2.imshow("matches", match_img)
cv2.waitKey(0)
