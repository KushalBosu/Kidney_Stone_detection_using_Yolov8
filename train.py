from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Train the model
model.train(
    data='kidney_stones.yaml', 
    epochs=100, 
    imgsz=512,
    batch=4,
    optimizer= "Adam",
    lr0=1e-3
)
