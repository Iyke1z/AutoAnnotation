import os
import sys
from os.path import *

root = os.path.join(os.path.dirname(__file__), "datasets")
coco_root = os.path.join(root, "coco2017/coco")
train_file = join(coco_root, "train", "train2017.txt")
val_file = join(coco_root, "val", "val2017.txt")

def convert_datasets_local(input_file):
    with open(input_file, "r") as fp:
        lines = fp.readlines()
        new_txt_ = "new"+os.path.basename(input_file)
        path_dirname = os.path.dirname(input_file)
        l = os.path.join(path_dirname, new_txt_)

        with open(l, "w") as fp_out:
            for line in lines:
                new_line = line.split("datasets")[1][1:]
                fp_out.write(new_line)
            fp_out.flush()
            fp_out.close()
        fp.flush()
        fp.close()

if __name__ == '__main__':
    convert_datasets_local(train_file)
    convert_datasets_local(val_file)