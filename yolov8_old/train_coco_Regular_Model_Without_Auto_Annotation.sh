#!/usr/bin/env bash
source /home/lamda/venv/bin/activate
python3 train.py --cfg coco_Regular_Model_Without_Auto_Annotation.yaml --data coco_Regular_Model_Without_Auto_Annotation.yaml --epochs 70 --weights ""  --batch-size 6
