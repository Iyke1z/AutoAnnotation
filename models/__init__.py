import torch

from models.deeplabv3plus import get_deeplabv3plus
from models.smp import get_smp
from models.yolov5 import get_yolov5, get_yolov5_nc7, get_yolov5_nc7_attention
from models.yolov6 import get_yolov6

MODELS = {
    # object detectors
    "yolov5": get_yolov5,
    "yolov6": get_yolov6,
    "yolov5_nc7": get_yolov5_nc7,
    "yolov5_nc7_attention":get_yolov5_nc7_attention

    # # segmentation models
    # "unet": get_smp("unet"),
    # "deeplabv3plus": get_deeplabv3plus,
}

def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    try:
        model = MODELS[model_name](**kwargs)
        return model
    except Exception as e:
        print(f"Model {model_name} not available")
        exit()
