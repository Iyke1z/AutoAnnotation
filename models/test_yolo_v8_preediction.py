from ultralytics import YOLO

yolo8_model_detection=YOLO('/home/projects/xai/exaplaiable_ai/runs/detect/train55/weights/best.pt')
yolo8_model_detection.predict(source='/home/projects/xai/exaplaiable_ai/data/', save=True )