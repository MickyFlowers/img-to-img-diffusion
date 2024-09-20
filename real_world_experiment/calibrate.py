import sys

sys.path.append("./")
from xlib.algo.calibrator import EyeHandCalibrator
from xlib.device.sensor.camera import RealSenseCamera
from xlib.device.manipulator.ur_robot import UR
import xlib.log

camera = RealSenseCamera()
ur_robot = UR(ip="192.168.1.102")
cali = EyeHandCalibrator(
    camera,
    ur_robot,
)
cali.setAruco()
cali.sampleImages("./calibration_data")
cali.calibrate("eye-in-hand", "./calibration_data")
