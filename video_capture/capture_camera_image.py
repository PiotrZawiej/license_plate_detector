import os

import cv2
import time
import threading

from fast_alpr import ALPR
from ultralytics import YOLO

from plate_from_iamge import extract_plate_from_image
from video_capture.check_plate import check_plate
from video_capture.trusted_plates_util import save_known_plates

model = YOLO(r'..\runs\detect\train3\weights\best.pt')
is_checking = None
last_check_time = None


def capture_plate():
    global is_checking, last_check_time

    print("🔍 Sprawdzam tablice...")
    time.sleep(3)

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        x, y, w, h = roi
        roi_frame = frame[y:y + h, x:x + w]

        try:
            plate_crop = extract_plate_from_image(roi_frame, model)

            alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model",
            )

            alpr_results = alpr.predict(plate_crop)
            if alpr_results:
                print("Odczytana wartosc tablicy: " + alpr_results[0].ocr.text)
                check_plate(alpr_results[0].ocr.text)
            else:
                print("❗ Tablica nie zostala rozpoznana.")
        except Exception as e:
            print(f"❌ Blad podczas rozpoznawania tablicy: {e}")
    else:
        print("❌ Nie udalo sie pobrac klatki.")
    is_checking = False
    last_check_time = time.time()


def start_capturing():
    global is_checking, last_check_time

    ret, prev_frame = cap.read()
    if not ret:
        print("Nie udalo sie odczytac z kamery.")
        cap.release()
        exit()

    prev_frame = cv2.resize(prev_frame, (WIDTH, HEIGHT))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    is_checking = False
    last_check_time = 0
    COOLDOWN_AFTER_FINISH = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = roi
        roi_prev = prev_gray[y:y + h, x:x + w]
        roi_current = gray[y:y + h, x:x + w]

        diff = cv2.absdiff(roi_prev, roi_current)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = any(cv2.contourArea(c) > min_area for c in contours)
        current_time = time.time()

        if movement_detected and not is_checking and (current_time - last_check_time > COOLDOWN_AFTER_FINISH):
            is_checking = True
            threading.Thread(target=capture_plate).start()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Kamera (16:9)", frame)

        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
roi = (420, 400, 400, 150)
threshold = 25
min_area = 500

save_known_plates()
start_capturing()

cap.release()
cv2.destroyAllWindows()
