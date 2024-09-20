import os
import shutil

data_root = "../data/real_world_data_4"
flist = []
fo = open(os.path.join(data_root, "val.flist"), "w")
for i in range(11600, 11700):
    img = f"{i:05d}"
    fo.write(img + ".jpg" + "\n")
