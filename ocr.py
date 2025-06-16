import cv2
import pytesseract
import numpy as np
import re

# Ścieżka do Tesseract (jeśli nie masz w PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"



def ocr(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Progowanie Otsu + offset
    # otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # adjusted_thresh = otsu_thresh + 10
    # _, binary = cv2.threshold(gray, adjusted_thresh, 255, cv2.THRESH_BINARY)

    # Przycinanie
    h, w = gray.shape
    top = int(h * 0.05)
    bottom = int(h * 0.85)
    left = int(w * 0.1)
    right = int(w * 0.97)
    cropped = gray[top:bottom, left:right]

    # OCR przez Tesseract
    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(cropped, config=config)
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    print(clean_text if clean_text else '[brak tekstu]')

    # Podgląd (opcjonalnie)
    # cv2.imshow("cropped", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return clean_text if clean_text else None
