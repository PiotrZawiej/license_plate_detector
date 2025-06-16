import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

image_folder = os.path.join("dataset", "train")
xml_path = os.path.join("dataset", "annotations.xml")
model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
model = YOLO(model_path)

tree = ET.parse(xml_path)
root = tree.getroot()
ground_truth = {}

for image in root.findall("image"):
    filename = image.get("name")
    box = image.find("box")
    if box is not None:
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))
        ground_truth[filename] = [int(xtl), int(ytl), int(xbr), int(ybr)]

ious = []
processed = 0

for file in os.listdir(image_folder):
    if not file.lower().endswith(".jpg"):
        continue

    if file not in ground_truth:
        print(f"⚠️ Brak danych GT w XML dla: {file}")
        continue

    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Nie można wczytać: {file}")
        continue

    gt_box = ground_truth[file]
    preds = model(img)[0].boxes.xyxy.cpu().numpy()

    if len(preds) == 0:
        print(f"❌ Brak detekcji: {file}")
        iou = 0.0
    else:
        pred_box = list(map(int, preds[0]))
        iou = compute_iou(pred_box, gt_box)

    ious.append(iou)
    processed += 1
    print(f"{file} | IoU: {iou:.3f}")

if processed > 0:
    mean_iou = sum(ious) / processed
    print(f"\nŚrednie IoU: {mean_iou:.3f} ({processed} obrazów)")
else:
    print("\n⚠️ Nie przetworzono żadnych obrazów.")
