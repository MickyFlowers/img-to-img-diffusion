from . import kp_matcher
from ..sensor.camera import Camera
import numpy as np
from . import metric
import cv2


class IBVS:
    def __init__(
        self,
        camera: Camera,
        kp_extractor=kp_matcher.KpMatchAlgo,
    ) -> None:
        self.kp_extractor = kp_extractor()
        self.camera = camera

    def cal_vel_from_kp(self, ref_kp, cur_kp, ref_z, cur_z):

        assert ref_kp.shape == cur_kp.shape, "Keypoints shape mismatch"
        ref_kp = self.camera.pixel_to_camera_frame(ref_kp)
        cur_kp = self.camera.pixel_to_camera_frame(cur_kp)
        num_kp = ref_kp.shape[0]

        cur_x = cur_kp[:, 0]
        cur_y = cur_kp[:, 1]

        cur_L = np.zeros((num_kp * 2, 6), cur_kp.dtype)
        cur_L[0::2, 0] = -1.0 / cur_z
        cur_L[0::2, 2] = cur_x / cur_z
        cur_L[0::2, 3] = cur_x * cur_y
        cur_L[0::2, 4] = -(1 + cur_x * cur_x)
        cur_L[0::2, 5] = cur_y
        cur_L[1::2, 1] = -1.0 / cur_z
        cur_L[1::2, 2] = cur_y / cur_z
        cur_L[1::2, 3] = 1 + cur_y * cur_y
        cur_L[1::2, 4] = -cur_x * cur_y
        cur_L[1::2, 5] = -cur_x

        ref_x = ref_kp[:, 0]
        ref_y = ref_kp[:, 1]

        ref_L = np.zeros((num_kp * 2, 6), ref_kp.dtype)
        ref_L[0::2, 0] = -1.0 / ref_z
        ref_L[0::2, 2] = ref_x / ref_z
        ref_L[0::2, 3] = ref_x * ref_y
        ref_L[0::2, 4] = -(1 + ref_x * ref_x)
        ref_L[0::2, 5] = ref_y
        ref_L[1::2, 1] = -1.0 / ref_z
        ref_L[1::2, 2] = ref_y / ref_z
        ref_L[1::2, 3] = 1 + ref_y * ref_y
        ref_L[1::2, 4] = -ref_x * ref_y
        ref_L[1::2, 5] = -ref_x

        error = np.zeros(num_kp * 2, cur_kp.dtype)
        error[0::2] = ref_x - cur_x
        error[1::2] = ref_y - cur_y

        mean_L = (cur_L + ref_L) / 2.0
        vel = np.linalg.lstsq(mean_L, error)[0]

        return vel

    def cal_vel_from_img(
        self, ref_img, cur_img, ref_depth, cur_depth, mask=None, use_median_depth=False
    ):
        assert (
            self.kp_extractor is not None and self.camera is not None
        ), "KeyPoint Extractor or Camera not provided"
        ref_kp, cur_kp, match_img = self.kp_extractor.match(
            ref_img, cur_img, mask, True, self.camera
        )
        ref_kp_int = ref_kp.round().astype(int)
        cur_kp_int = cur_kp.round().astype(int)
        ref_kp_x = ref_kp_int[:, 0]
        ref_kp_y = ref_kp_int[:, 1]
        cur_kp_x = cur_kp_int[:, 0]
        cur_kp_y = cur_kp_int[:, 1]

        ref_depth = ref_depth.squeeze()
        cur_depth = ref_depth.squeeze()

        ref_z = ref_depth[ref_kp_y, ref_kp_x]
        cur_z = cur_depth[cur_kp_y, cur_kp_x]
        if use_median_depth:
            ref_z = np.median(ref_z)
            cur_z = np.median(cur_z)

        vel = self.cal_vel_from_kp(ref_kp, cur_kp, ref_z, cur_z)
        score = metric.calc_ssim(ref_img, cur_img)
        cv2.putText(
            match_img,
            "SSIM score: {:.3f}".format(score),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        return vel, score, match_img
