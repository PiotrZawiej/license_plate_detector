import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    # Intersection over Union between two [x1,y1,x2,y2] boxes
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: 
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

# Paths and model load
image_folder = os.path.join("dataset", "train")
xml_path     = os.path.join("dataset", "annotations.xml")
model = YOLO(os.path.join("runs", "detect", "train", "weights", "best.pt"))

# Parse ground-truth boxes from XML
def calculate_iou():
    ground_truth = {}
    tree = ET.parse(xml_path)
    for img in tree.getroot().findall("image"):
        box = img.find("box")
        if box is not None:
            ground_truth[img.get("name")] = [
                int(float(box.get("xtl"))),
                int(float(box.get("ytl"))),
                int(float(box.get("xbr"))),
                int(float(box.get("ybr"))),
            ]

    ious, processed = [], 0
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(".jpg") or fname not in ground_truth:
            print(f"⚠️ Skipping: {fname}")
            continue

        img = cv2.imread(os.path.join(image_folder, fname))
        if img is None:
            print(f"❌ Cannot load: {fname}")
            continue

        preds = model(img)[0].boxes.xyxy.cpu().numpy()
        if len(preds) == 0:
            print(f"❌ No detection: {fname}")
            iou = 0.0
        else:
            iou = compute_iou(list(map(int, preds[0])), ground_truth[fname])

        ious.append(iou)
        processed += 1
        print(f"{fname} | IoU: {iou:.3f}")

    if processed:
        print(f"\nMean IoU: {sum(ious)/processed:.3f} over {processed} images")
    else:
        print("\n⚠️ No images processed.")

    return sum(ious)/processed
