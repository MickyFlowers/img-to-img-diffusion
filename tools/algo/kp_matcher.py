import cv2
import cv2
import numpy as np
from ..sensor.camera import Camera


class KpMatchAlgo:
    def __init__(self, kp_extractor: str = "SIFT", match_threshold=0.75) -> None:
        self.match_threshold = match_threshold
        self.kp_extractor = self._parser_kp_extractor(kp_extractor)
        self.matcher = cv2.FlannBasedMatcher()

    def _parser_kp_extractor(self, kp_extractor_str: str = "SIFT"):
        kp_extractor = "cv2." + kp_extractor_str.upper() + "_create"
        try:
            kp_extractor = eval(kp_extractor)
        except:
            raise ValueError("Invalid keypoint extractor in opencv")
        return kp_extractor()

    def _kp_extract(self, img1: np.ndarray, img2: np.ndarray):
        assert img1 is not None, "Color Image 1 not provided"
        assert img2 is not None, "Color Image 2 not provided"
        kp1, des1 = self.kp_extractor.detectAndCompute(img1, None)
        kp2, des2 = self.kp_extractor.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2

    def match(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask=None,
        ransac: bool = True,
        camera: Camera = None,
    ):
        assert img1 is not None, "Color Image 1 not provided"
        assert img2 is not None, "Color Image 2 not provided"
        kp1, des1, kp2, des2 = self._kp_extract(img1, img2)
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_threshold * n.distance:
                good_matches.append([m])

        if len(good_matches) < 3:
            print("Too few matches :{}".format(len(good_matches)))
            return None, None, None
        kp1_array = np.array(
            [kp1[good_matches[i][0].queryIdx].pt for i in range(len(good_matches))]
        )

        kp2_array = np.array(
            [kp2[good_matches[i][0].trainIdx].pt for i in range(len(good_matches))]
        )
        if mask is not None:
            kp1_array_int = kp1_array.round().astype(int)
            kp2_array_int = kp2_array.round().astype(int)

            x1, y1 = kp1_array_int[:, 0], kp1_array_int[:, 1]
            x2, y2 = kp2_array_int[:, 0], kp2_array_int[:, 1]
            mask = mask[y1, x1] & mask[y2, x2]
            mask = ~mask
            kp1_array = kp1_array[mask]
            kp2_array = kp2_array[mask]
            mask_indices = np.where(mask)[0]
            good_matches = [good_matches[i] for i in mask_indices]

        if len(good_matches) > 4 and ransac:
            _, mask = cv2.findEssentialMat(
                kp1_array.reshape(-1, 1, 2),
                kp2_array.reshape(-1, 1, 2),
                camera.intrinsic,
                cv2.RANSAC,
                0.999,
                3.0,
            )
            mask = mask.ravel().astype(bool)
            kp1_array = kp1_array[mask].reshape(-1, 2)
            kp2_array = kp2_array[mask].reshape(-1, 2)
            mask_indices = np.where(mask)[0]
            good_matches = [good_matches[i] for i in mask_indices]

        match_img = cv2.drawMatchesKnn(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

        cv2.putText(
            match_img,
            f"Matched Features: {len(good_matches)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        return kp1_array, kp2_array, match_img
