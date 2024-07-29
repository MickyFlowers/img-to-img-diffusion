import skimage.metrics
import numpy as np

def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    assert img1.shape == img2.shape, "Image shapes do not match"
    score = skimage.metrics.structural_similarity(img1, img2, channel_axis=2)
    return score
