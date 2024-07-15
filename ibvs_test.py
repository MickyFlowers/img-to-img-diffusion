from tools.ibvs import IBVS
from tools.perception import Frame, PinholeCamera
import cv2
import numpy as np

camera = PinholeCamera(
    width=640,
    height=480,
    fx=616.56402588,
    fy=616.59606934,
    cx=330.48983765,
    cy=233.84162903,
)
ref_img = cv2.imread("/home/chenyuxi/project/img-to-img-diffusion/test_img/ref_img.png")
cur_img = cv2.imread("/home/chenyuxi/project/img-to-img-diffusion/test_img/img.png")
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
mask = np.load("/home/chenyuxi/project/img-to-img-diffusion/test_img/mask.npy")
cur_img[mask] = 0
ref_img[mask] = 0
policy = IBVS()
cur_depth = np.full([480, 640, 1], 0.3)
tar_depth = np.full([480, 640, 1], 0.2)
wcT = np.eye(4)
cur_frame = Frame(
    camera=camera,
    color=cur_img,
    depth=cur_depth,
    seg=None,
    wcT=wcT
)
tar_frame = Frame(
    camera=camera,
    color=ref_img,
    depth=tar_depth,
    seg=None,
    wcT=wcT
)
policy.set_desired(tar_frame)
vel, aux = policy.compute_velocity(cur_frame)
print(vel)
cv2.imshow("desired | current", aux["plottings"])
cv2.waitKey(0) & 0xFF == ord("q")