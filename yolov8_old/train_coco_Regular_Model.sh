#!/usr/bin/env bash
source /home/lamda/venv/bin/activate
python3 train.py --cfg coco_Regular_Model.yaml --data coco_Regular_Model.yaml --epochs 100 --weights ""  --batch-size 6
