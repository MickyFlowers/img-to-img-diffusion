import os
import shutil

data_root = "../data/ddpm_visual_servo/img"
flist = []
fo = open(os.path.join(data_root, "train.flist"), "w")
for i in range(100000):
    img = "img-{}".format(i)
    fo.write(img + "\n")
