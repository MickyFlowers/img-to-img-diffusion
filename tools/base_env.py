import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_env.non_isaac.action import action
from isaac_env.non_isaac import sampling


def close_enough(T0: np.ndarray, T1: np.ndarray, du_eps: float, dt_eps: float):
    """
    - T0, T1: two poses to be evaluated, 4x4 transformation matrix
    - du_eps: float, unit: rad
    - dt_eps: float, unit: meter
    """
    dT = np.linalg.inv(T0) @ T1
    du = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec())
    dt = np.linalg.norm(dT[:3, 3])
    return (du < du_eps) and (dt < dt_eps)


def print_pose_err(T0: np.ndarray, T1: np.ndarray):
    dT = np.linalg.inv(T0) @ T1
    du = R.from_matrix(dT[:3, :3]).as_rotvec()
    dt = dT[:3, 3]

    du_deg = np.linalg.norm(du) / np.pi * 180
    dt_mm = np.linalg.norm(dt) * 1000
    print("[INFO] Pose error: |du| = {:.3f}°, |dt| = {:.3f}mm"
          .format(du_deg, dt_mm))
    return du_deg, dt_mm


class BaseEnv(object):
    # parameters used for desired pose sampling
    desired_pose_r = [0.5, 0.9]
    desired_pose_phi = [60, 90]
    desired_pose_drz_max = 15
    desired_pose_dry_max = 5
    desired_pose_drx_max = 5

    # parameters used for initial pose sampling
    initial_pose_r = [0.5, 0.9]
    initial_pose_phi = [30, 90]
    initial_pose_drz_max = 60
    initial_pose_dry_max = 10
    initial_pose_drx_max = 10

    # parameters used for initial pose sampling 
    # (apply right-hand transformation from desired pose)
    pose_disturb_drz = 10
    pose_disturb_dry = 10
    pose_disturb_drx = 10
    pose_disturb_dt = 0.02

    dt = 0.02  # 50Hz
    angle_eps_degree = 1  # degree
    dist_eps_mm = 2  # mm
    max_steps = 400
    
    def __init__(
        self, 
        resample=True, 
        auto_reinit=False, 
        verbose=False
    ):
        # initialize option
        self.first = True
        self.resample = resample
        self.auto_reinit = auto_reinit
        self.verbose = verbose

        # state
        self._current_wcT = None
        self.desired_wcT = None
        self.initial_wcT = None
        self.steps = 0
    
    @property
    def current_wcT(self):
        return self._current_wcT
    
    @current_wcT.setter
    def current_wcT(self, cur_wcT: np.ndarray):
        self._current_wcT = cur_wcT
    
    def initialize_scene(self):
        raise NotImplementedError
    
    def initialize(self, disturb_from_desired=False):
        """This function will call `initialize_scene` to prepare for scene, 
        and sample the desired pose and initial pose.

        Arguments: 
        - disturb_from_desired: bool, if True, the initial pose is disturbed from 
            the desired pose, otherwise directly sampled from sampling space 
        """
        if self.resample or self.first:
            self.initialize_scene()
            self.desired_wcT = self.sample_desired_pose()
            if not disturb_from_desired:
                self.initial_wcT = self.sample_initial_pose()
            else:
                self.initial_wcT = self.desired_wcT @ self.sample_pose_disturbance()
        self.current_wcT = self.initial_wcT.copy()
        self.steps = 0
        self.first = False

    def observation(self):
        raise NotImplementedError

    def action(self, vel: np.ndarray):
        self.current_wcT = action(self.current_wcT, vel, self.dt)
        self.steps += 1

        if self.auto_reinit and self.need_reinit():
            self.initialize()
    
    def sample_desired_pose(self):
        return sampling.sample_camera_pose(
            r_min=self.desired_pose_r[0],
            r_max=self.desired_pose_r[1],
            phi_min=self.desired_pose_phi[0],
            phi_max=self.desired_pose_phi[1],
            drz_max=self.desired_pose_drz_max,
            dry_max=self.desired_pose_dry_max,
            drx_max=self.desired_pose_drx_max
        )
    
    def sample_initial_pose(self):
        return sampling.sample_camera_pose(
            r_min=self.initial_pose_r[0],
            r_max=self.initial_pose_r[1],
            phi_min=self.initial_pose_phi[0],
            phi_max=self.initial_pose_phi[1],
            drz_max=self.initial_pose_drz_max,
            dry_max=self.initial_pose_dry_max,
            drx_max=self.initial_pose_drx_max
        )
    
    def sample_pose_disturbance(self):
        return sampling.sample_pose_disturb(
            drz_max=self.pose_disturb_drz,
            dry_max=self.pose_disturb_dry,
            drx_max=self.pose_disturb_drx,
            dt_max=self.pose_disturb_dt
        )
    
    def close_enough(self):
        return close_enough(self.desired_wcT, self.current_wcT, 
                            self.angle_eps_degree / 180 * np.pi,
                            self.dist_eps_mm / 1000)

    def exceeds_maximum_steps(self):
        return self.steps > self.max_steps

    def abnormal_pose(self):
        pos = self.current_wcT[:3, 3]
        if pos[-1] < 0:
            if self.verbose:
                print("[INFO] Camera under ground")
            return True
        
        zaxis = self.current_wcT[:3, 2]
        if zaxis[-1] > -0.1:
            if self.verbose:
                print("[INFO] Camera almost look up")
            return True

        dist = np.linalg.norm(pos)
        if dist > self.desired_pose_r[-1] * 2:
            if self.verbose:
                print("[INFO] Too far away")
            return True
        
        return False
    
    def need_reinit(self):
        if self.close_enough():
            return True
        
        if self.exceeds_maximum_steps():
            if self.verbose:
                print("[INFO] Overceed maximum steps")
            return True
        
        if self.abnormal_pose():
            return True
        
        return False

    def print_pose_err(self):
        return print_pose_err(self.current_wcT, self.desired_wcT)

