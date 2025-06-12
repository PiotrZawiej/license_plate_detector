import os
import cv2
import xml.etree.ElementTree as ET
import time

from plate_from_iamge import extract_plate_from_image
from ultralytics import YOLO
from fast_alpr import ALPR

model_path = r'runs\detect\train3\weights\best.pt'
image_folder = r'dataset\test'
xml_path = r'dataset\annotations.xml'

for file in os.listdir(image_folder):
    if file.endswith('.png') or file.endswith('.txt'):
        try:
            os.remove(os.path.join(image_folder, file))
            print(f"deleted: {file}")
        except Exception as e:
            print(f"error {file}: {e}")

# === Wczytywanie modeli ===
model = YOLO(model_path)
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

tree = ET.parse(xml_path)
root = tree.getroot()

ground_truth = {}
for image in root.findall('image'):
    filename = image.get('name')
    box = image.find('box')
    if box is not None:
        plate_number = box.find('attribute').text.strip().upper()
        ground_truth[filename] = plate_number

total = 0
correct = 0
start_time = time.time()

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_folder, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue

    plate_crop = extract_plate_from_image(image, model)
    prediction = None

    if plate_crop is not None:
        try:
            results = alpr.predict(plate_crop)
            if results:
                prediction = results[0].ocr.text.strip().upper()
        except Exception as e:
            print(f"Fstalpr error {filename}: {e}")
    else:
        print(f"no detection: {filename}")

    expected = ground_truth.get(filename, None)

    if expected and prediction:
        total += 1
        if expected == prediction:
            correct += 1
        else:
            print(f"❌ {filename} | OCR: {prediction} | GT: {expected}")
    elif expected:
        total += 1
        print(f"⚠️ {filename} | no detection | GT: {expected}")

# === Podsumowanie ===
end_time = time.time()
accuracy = (correct / total) * 100 if total else 0
elapsed = end_time - start_time

print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"\nCzas działania: {elapsed:.2f} s (~{elapsed/60:.2f} min)")
