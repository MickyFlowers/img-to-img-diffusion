import sys

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR
from xlib.device.sensor.camera import RealSenseCamera
import logging
import xlib.log
import numpy as np
import cv2

ur_robot: UR = UR("192.168.1.102")
camera = RealSenseCamera()
while True:
    camera.get_frame()
    camera.show()
    key = cv2.waitKey(1)
    if key == ord("p"):
        pre_tcp_pose = ur_robot.tcp_pose
        logging.info(f"pre_tcp_pose---------------: \n {repr(pre_tcp_pose)}")
        np.save("real_world_experiment/tcp_pose/pre_tcp_pose.npy", pre_tcp_pose)
    elif key == ord("g"):
        execute_tcp_pose = ur_robot.tcp_pose
        logging.info(f"execute_tcp_pose---------------: \n {repr(execute_tcp_pose)}")
        np.save("real_world_experiment/tcp_pose/execute_tcp_pose.npy", execute_tcp_pose)
    elif key == ord("q"):
        try:
            relative_pose = np.linalg.inv(pre_tcp_pose) @ execute_tcp_pose
            np.save("real_world_experiment/tcp_pose/relative_pose.npy", relative_pose)
        except ValueError as e:
            logging.error("pre_tcp_pose and execute_tcp_pose is None")
            logging.exception(e)
        break
ur_robot.disconnect()
