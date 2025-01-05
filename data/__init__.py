from typing import Dict, Any

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

from datasets.coco2017 import coco2017_train, coco2017_test
from datasets.cityscapes import cityscapes_train, cityscapes_test
from datasets.voc2012 import voc2012_train, voc2012_test

DATASETS = {
    "coco2017":
        {"train": coco2017_train,
         "test": coco2017_test,
         "n_classes": 7},
    "coco_yolo":
        {"train": coco2017_train,
         "test": coco2017_test,
         "n_classes": 7},
    # "cityscapes": 
    #     {"train": cityscapes_train,
    #      "test": cityscapes_test,
    #      "n_classes": 20},
    # "voc2012":
    #     {"train": voc2012_train,
    #      "test": voc2012_test,
    #      "n_classes": 21},
}


def get_dataset(dataset_name: str) -> Dict[str, Any]:
    print("INIT", dataset_name)
    try:
        dataset = DATASETS[dataset_name]
        return dataset
    except KeyError:
        print(f"DATASET {dataset_name} not defined.")
        exit()


def get_sample(dataset: Dataset, sample_id: int, device):
    # get sample and push it to device
    data = dataset[sample_id]
    processed = []
    for x in data:
        # print(type(x))
        if isinstance(x, torch.Tensor) and len(x.shape):
            processed.append(x[None, :].to(device))
        elif isinstance(x, int) or isinstance(x, np.int) or isinstance(x, torch.Tensor) or isinstance(x, np.int64):
            processed.append(torch.Tensor([x])[None, :].to(device))
        else:
            print(f"data sample of type {type(x)} not put to device.")
            if "labels" in x:
                processed.append(torch.Tensor(x['labels']))
            else:
                print(x)
    return processed
