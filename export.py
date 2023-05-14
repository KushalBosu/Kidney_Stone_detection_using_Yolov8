from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt') 

# Export the model
model.export()
