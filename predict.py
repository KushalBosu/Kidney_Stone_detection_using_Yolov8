from ultralytics import YOLO

model= YOLO("best.pt")

model.predict("stone4.jpg",save=True,imgsz=512,conf=0.5)