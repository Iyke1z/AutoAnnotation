#!/usr/bin/env bash
source /home/lamda/venv/bin/activate
python3 train.py --cfg coco_Attention_Model.yaml --data coco_Attention_Model.yaml --epochs 80 --weights ""  --batch-size 128
