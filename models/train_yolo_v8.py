from ultralytics import YOLO
import sys
import os

dirname = os.path.abspath(os.path.dirname(__file__))
# Load a model
model = YOLO('yolov8m.yaml', task="detect")  # build a new model from YAML
model = YOLO('yolov8m.pt', task="detect")  # load a pretrained model (recommended for training)
model = YOLO('yolov8m.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data= os.path.join(dirname,'coco_yolo.yaml'), epochs=1, imgsz=640)
