from ultralytics import YOLO
import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('runs/detect/train2/weights/best.pt')

img_path = r'test_photos1\94c5a151-24b5-493c-a900-017a4353b00c___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Tata-Tiago-Front-Number-Plates-Design.jpg'
image = cv2.imread(img_path)

results = model(image)

for r in results:
    for i, box in enumerate(r.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plate = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized_plate = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(resized_plate, config=config)

        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if clean_text:
            print(f"[{i}] Odczytana tablica: {clean_text}")
        else:
            print(f"[{i}] Nie udało się odczytać tekstu z tablicy.")

image = cv2.resize(image, (1000, 640))

cv2.imshow("wynik.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
