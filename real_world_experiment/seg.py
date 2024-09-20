import sys

sys.path.append("./")

from xlib.sam.sam_gui import SAM

sam = SAM("../data/real_world_data_3/img", "../data/real_world_data_3/first_seg")
sam.segment()
