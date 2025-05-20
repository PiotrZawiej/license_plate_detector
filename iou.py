import os
import cv2
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def txt_to_box(txt_path, img_shape):
    h, w = img_shape[:2]
    with open(txt_path) as f:
        xc, yc, bw, bh = map(float, f.readline().split()[1:])
    x1 = int((xc - bw/2) * w)
    y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w)
    y2 = int((yc + bh/2) * h)
    return [x1, y1, x2, y2]

image_folder = "test_photos"
label_folder = r"dataset\labels"
model = YOLO("runs/detect/train2/weights/best.pt")
ious = []

for file in os.listdir(image_folder):
    if not file.endswith(".jpg"):
        continue

    img_path = os.path.join(image_folder, file)
    label_path = os.path.join(label_folder, file.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    gt_box = txt_to_box(label_path, img.shape)
    pred = model(img)[0].boxes.xyxy.cpu().numpy()

    if len(pred) == 0:
        ious.append(0.0)
    else:
        pred_box = list(map(int, pred[0]))  
        ious.append(compute_iou(pred_box, gt_box))

print(f"Średnie IoU: {sum(ious)/len(ious):.3f}")
