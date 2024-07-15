from tools.perception import Frame, PinholeCamera
from tools.frontend.classic import Classic as ClassicFrontend
from tools import classic_visual_servo
from tools.frontend.utils import plot_corr
import numpy as np

class BasePolicy(object):
    def set_desired(self, frame: Frame):
        raise NotImplementedError
    
    def compute_velocity(self, frame: Frame):
        raise NotImplementedError
    
class IBVS(BasePolicy):
    def __init__(self, use_median_depth: bool = False):
        super().__init__()
        self.frontend = ClassicFrontend(
            intrinsic=None,
            detector="SIFT:0",
            ransac=True,
        )
        self.desired_frame = None
        self.use_median_depth = use_median_depth
    
    def set_desired(self, frame: Frame):
        self.desired_frame = frame
        bgr = np.ascontiguousarray(frame.color[:, :, [2, 1, 0]])
        self.frontend.intrinsic = (
            frame.camera if isinstance(frame.camera, PinholeCamera) else
            frame.camera.intrinsic)
        self.frontend.update_target_frame(bgr)
    
    def compute_velocity(self, frame: Frame):
        bgr = np.ascontiguousarray(frame.color[:, :, [2, 1, 0]])
        corr = self.frontend.process_current_frame(bgr)
        H, W = bgr.shape[:2]

        fp_tar: np.ndarray = corr.tar_pos[corr.valid_mask]  # (N, 2)
        fp_cur: np.ndarray = corr.cur_pos_aligned[corr.valid_mask]  # (N, 2)

        assert (fp_tar.max(axis=0) < np.array([W, H])).all()
        assert (fp_cur.max(axis=0) < np.array([W, H])).all()

        # get per-pixel depth
        cc_tar, rr_tar = np.round(fp_tar).astype(np.int32).clip(0, [W-1, H-1]).T
        cc_cur, rr_cur = np.round(fp_cur).astype(np.int32).clip(0, [W-1, H-1]).T
        Z_tar = self.desired_frame.depth[rr_tar, cc_tar]
        Z_cur = frame.depth[rr_cur, cc_cur]

        if self.use_median_depth:
            Z_tar = np.median(Z_tar)
            Z_cur = np.median(Z_cur)
        # Z_tar = np.linalg.norm(self.desired_frame.wcT[:3, 3])
        # Z_cur = np.linalg.norm(frame.wcT[:3, 3])

        vel = classic_visual_servo.match_based.ibvs(
            fp_cur=fp_cur, Z_cur=Z_cur,
            fp_tar=fp_tar, Z_tar=Z_tar,
            intrinsic=self.frontend.intrinsic
        )
        aux = {
            "plottings": plot_corr(corr)
        }
        return vel, aux