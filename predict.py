from ultralytics import YOLO

model= YOLO("runs/detect/train5/weights/best.pt")

model.predict("stone4.jpg",save=True,imgsz=512,conf=0.5)