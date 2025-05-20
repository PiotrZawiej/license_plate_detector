import os
import cv2
from ultralytics import YOLO
import time

MIN_WIDTH = 300
MIN_HEIGHT = 300

model = YOLO("runs/detect/train2/weights/best.pt")
image_dir = "test_photos1"
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

correct = 0
total = 0

start_time = time.time()

for filename in image_files:
    img_path = os.path.join(image_dir, filename)
    image = cv2.imread(img_path)

    h, w = image.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        continue

    results = model(image)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(pred_boxes) > 0:
        correct += 1

    total += 1

end_time = time.time()

accuracy = correct / total if total > 0 else 0
processing_time = end_time - start_time
avg_time_per_image = processing_time / total if total > 0 else 0


print(f"Przetworzono: {total} obrazów")
print(f"Accuracy: {accuracy:.2%}")
print(f"Łączny czas przetwarzania: {processing_time:.2f} sekundy")
