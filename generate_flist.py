import os
import shutil

data_root = "/cyx/data/vs/fixed_relative_pose_img-2"
flist = []
fo = open(os.path.join(data_root, "train.flist"), "w")
for i in range(100000):
    img = "img-{}".format(i)
    fo.write(img + "\n")
