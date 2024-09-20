import cv2
import os
from PIL import Image

imgs_path = "../data/real_world_data_4/img"
output_path = "../data/real_world_data_4/down_sample_img"

img_num = len(os.listdir(imgs_path))
for i in range(len(os.listdir(imgs_path))):
    img = cv2.imread(os.path.join(imgs_path, f"{i:05d}.jpg"))
    down_sampling_img = cv2.resize(img, (320, 240))
    cv2.imwrite(os.path.join(output_path, f"{i:05d}.jpg"), down_sampling_img)
