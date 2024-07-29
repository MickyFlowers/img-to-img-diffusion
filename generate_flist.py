import os
import shutil

data_root = "../data/real_world_data"
flist = []
fo = open(os.path.join(data_root, "train_debug.flist"), "w")
for i in range(16):
    img = "img-{}".format(i)
    fo.write(img + "\n")
