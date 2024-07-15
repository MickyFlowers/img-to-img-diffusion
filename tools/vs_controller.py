import numpy as np
from scipy.spatial.transform import Rotation as R

class PBVSNumpy:
    def __init__(self, T_w_dc=np.zeros((4,4)),T_w_c=np.zeros((4,4))):
        self.T_w_dc = T_w_dc
        self.T_w_c = T_w_c
        self.vel = np.ones((6,))

    def cal_action(self, T_w_dc, T_w_c):
        self.T_w_dc = T_w_dc
        self.T_w_c = T_w_c
        self.T_dc_c = np.linalg.inv(self.T_w_dc).dot(self.T_w_c)
        RT = self.T_dc_c[:3,:3].T
        t_dc_c = self.T_dc_c[:3,3]
        thetau = R.from_matrix(self.T_dc_c[:3, :3]).as_rotvec()
        self.vel = np.ones((6,))
        self.vel[:3] = -0.5 * np.dot(RT, t_dc_c)
        self.vel[3:] = -0.5 * thetau
        return self.vel

    def cal_action_curve(self,cur_wcT, tar_wcT, points):
        wPo = np.mean(points, axis=0)  # w: world frame, o: center, P: points
        tar_cwT = np.linalg.inv(tar_wcT)
        tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
        # t: target camera frame, c: current camera frame, w: world frame

        cur_cwT = np.linalg.inv(cur_wcT)
        cPo = wPo @ cur_cwT[:3, :3].T + cur_cwT[:3, 3]

        tcT = tar_cwT @ cur_wcT
        u = R.from_matrix(tcT[:3, :3]).as_rotvec()

        v = -(tPo - cPo + np.cross(cPo, u))
        w = -u
        vel = np.concatenate([v, w])

        tPo_norm = np.linalg.norm(tPo)
        vel_si = np.concatenate([v / (tPo_norm + 1e-7), w])

        return vel, (tPo_norm, vel_si)

class IBVSNumpy:
    def __init__(self, feature_points=5, is_Le = False):
        self.feature_points = feature_points
        self.L = np.ones((feature_points * 2, 6))
        self.pL = np.linalg.pinv(self.L)
        self.goal = np.ones((feature_points, 2))
        self.state = np.ones((feature_points, 2))
        self.error = self.state.reshape(feature_points * 2, ) \
                     - self.goal.reshape(feature_points * 2, )
        self.is_Le = is_Le

    def updateL(self):
        Z = 0.2
        f = 616
        for i in range(self.feature_points):
            x = self.state[i, 0]
            y = self.state[i, 1]
            # Z = self.goal_distance[i]
            self.L[i * 2:i * 2 + 2, :] = np.array(
                [[-f / Z, 0, x / Z, x * y / f, -(f * f + x * x) / f, y],
                 [0, -f / Z, y / Z, (f * f + y * y) / f, -x * y / f, -x]])
        # print('Cond L:{}'.format(np.linalg.cond(self.L)))
        self.pL = np.linalg.pinv(self.L)
        self.error = self.state.reshape(self.feature_points * 2, ) \
                     - self.goal.reshape(self.feature_points * 2, )
        err_x = np.mean(abs(self.error[::2]))
        err_y = np.mean(abs(self.error[1::2]))
        # print('err_x:{}'.format(err_x))
        # print('err_y:{}'.format(err_y))

    def updateLe(self):
        Z = 0.2
        f = 616
        for i in range(self.feature_points):
            x = self.goal[i, 0]
            y = self.goal[i, 1]
            # Z = self.goal_distance[i]
            self.L[i * 2:i * 2 + 2, :] = np.array(
                [[-f / Z, 0, x / Z, x * y / f, -(f * f + x * x) / f, y],
                 [0, -f / Z, y / Z, (f * f + y * y) / f, -x * y / f, -x]])

        self.pL = np.linalg.pinv(self.L)
        self.error = self.state.reshape(self.feature_points * 2, ) \
                     - self.goal.reshape(self.feature_points * 2, )

    def cal_action(self, goal_points, state_points):
        self.state = state_points #* np.array([640.,480.])
        self.goal = goal_points #* np.array([640.,480.])
        if self.is_Le:
            self.updateLe()
        else:
            self.updateL()
        vel = -0.5 * np.dot(self.pL, self.error)
        return vel