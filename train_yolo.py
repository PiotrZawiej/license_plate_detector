from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Trenuj
model.train(
    data='data.yaml',
    epochs=20,
    imgsz=640
)
