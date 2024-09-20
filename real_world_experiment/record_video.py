import sys

sys.path.append("./")

from xlib.device.sensor.camera import RealSenseCamera
import cv2

camera = RealSenseCamera(exposure_time=500)
import time

time.sleep(2)
camera.recordVideo("../data/real_world_data_4/video/video.avi")
camera.stop()
