from ultralytics import YOLO
import cv2
import easyocr
import re

model = YOLO('runs/detect/train2/weights/best.pt')
reader = easyocr.Reader(['pl'])

img_path = r'test_photos\car_test2.jpg'
image = cv2.imread(img_path)
image = cv2.resize(image, (1000, 640))

results = model(image)

for r in results:
    for i, box in enumerate(r.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plate = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized_plate = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        ocr_result = reader.readtext(resized_plate)
        if ocr_result:
            _, text, prob = ocr_result[0]
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            print(f"[{i}] Odczytana tablica: {clean_text} (pewność: {prob:.2f})")
        else:
            print(f"[{i}] Nie udało się odczytać tekstu z tablicy.")

cv2.imshow("wynik.jpg", image)
cv2.waitKey(0)