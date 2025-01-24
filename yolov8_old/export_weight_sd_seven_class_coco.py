import torch
import os
import sys

root_name = os.path.abspath(os.path.dirname(__file__))
exp_name = "exp140"
model = torch.load( os.path.join(root_name, "runs/train",exp_name, "weights/best.pt"))
model["model"].training = True
sd = model["model"].state_dict()
model_export_root = os.path.abspath(os.path.join(root_name, ".."))

torch.save(sd,  os.path.join(model_export_root, 'models/checkpoints/yolov5_nc7_sd.pt'))