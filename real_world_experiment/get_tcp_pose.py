import sys

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR

ur_robot = UR("192.168.1.102")
tcp_pose = ur_robot.tcp_pose
import numpy as np

np.save("real_world_experiment/tcp_pose/vs_tcp_pose.npy", tcp_pose)
