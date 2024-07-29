import cv2
import numpy as np
from tools.perception import Frame, PinholeCamera
from tools.ibvs import IBVS

image1_path = "/home/cyx/project/img-to-img-diffusion/test_img/ref_img.png"
image2_path = "/home/cyx/project/img-to-img-diffusion/test_img/img.png"
mask = np.load("/home/cyx/project/img-to-img-diffusion/test_img/mask.npy")
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

policy = IBVS()
pinhole_camera = PinholeCamera(
    width=640,
    height=480,
    fx=616.56402588,
    fy=616.59606934,
    cx=330.48983765,
    cy=233.84162903,
)
depth = np.full((480, 640), 0.5)
frame1 = Frame(
    camera=pinhole_camera,
    color=img1,
    depth=depth,
    seg=None,
    wcT=np.eye(4),
)

frame2 = Frame(
    camera=pinhole_camera,
    color=img2,
    depth=depth,
    seg=None,
    wcT=np.eye(4),
)
policy.set_desired(frame1)
vel, aux = policy.compute_velocity(frame2)
print(vel)
cv2.imshow("img1", aux["plottings"])
cv2.waitKey(0)

