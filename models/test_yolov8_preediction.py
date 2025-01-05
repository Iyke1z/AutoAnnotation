
import argparse
import math
import os
import platform
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from torch import nn
from torchinfo import summary

from models.yolov5 import Model

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# model = Model(cfg=os.path.join("/home/projects/xai/exaplaiable_ai/ALM/L-CRP/yolov8/data","coco_seven_class.yaml"))
# sd = torch.load("/home/projects/xai/exaplaiable_ai/ALM/L-CRP/yolov8/runs/train/exp3/weights/best.pt")
model = Model(cfg=os.path.join(ROOT_DIR, "coco_seven_class.yaml"), nc=7)
sd = torch.load(os.path.join(ROOT_DIR, "checkpoints/yolov8_nc7_sd.pt"))
model.load_state_dict(sd)
model_stats = summary(model, (1, 3, 28, 28), verbose=0)
summary_str = str(model_stats)
print(summary_str)

