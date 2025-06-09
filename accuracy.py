import os
import cv2
import xml.etree.ElementTree as ET
import time  # ⏱️ Dodany import

from plate_from_iamge import extract_plate_from_image
from ocr import ocr
from ultralytics import YOLO

model = YOLO(r'runs\detect\train3\weights\best.pt')
image_folder = r'dataset\test'
xml_path = r'dataset\annotations.xml'

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

    prediction = None
    try:
        prediction = ocr(extract_plate_from_image(image, model))
    except:
        pass

    expected = ground_truth.get(filename, None)

    if expected and prediction:
        total += 1
        if expected == prediction:
            correct += 1
        else:
            print(f"❌ {filename} | OCR: {prediction} | GT: {expected}")
    elif expected:
        total += 1
        print(f"⚠️ {filename} | Brak predykcji | GT: {expected}")

end_time = time.time()  

accuracy = (correct / total) * 100 if total else 0
elapsed = end_time - start_time
print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"time: {elapsed:.2f} s (~{elapsed/60:.2f} min)")
