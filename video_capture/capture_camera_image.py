import os
import cv2
import time
import threading
import sys

# allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fast_alpr import ALPR
from plate_from_iamge import extract_plate_from_image
from ultralytics import YOLO

from video_capture.check_plate import check_plate
from video_capture.trusted_plates_util import save_known_plates

# Model and camera settings
model_path = os.path.join('train3', 'weights', 'best.pt')
model = YOLO(model_path)

# Globals to manage cooldown between checks
is_checking = False
last_check_time = None

def capture_plate():
    """
    Grab a frame, detect the plate region, run OCR, and check against trusted list.
    This runs in a separate thread when movement is detected.
    """
    global is_checking, last_check_time

    print("🔍 Checking plates...")
    time.sleep(3)  # small delay before capture

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        is_checking = False
        return

    # Crop to region of interest
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]

    try:
        # Detect plate and crop image
        plate_crop = extract_plate_from_image(roi_frame, model)

        # Initialize ALPR (detector + OCR)
        alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model",
        )

        # Perform OCR on the cropped plate
        results = alpr.predict(plate_crop) if plate_crop is not None else []
        if results:
            text = results[0].ocr.text.strip()
            print(f"Detected plate value: {text}")
            check_plate(text)
        else:
            print("❗ Plate was not recognized.")
    except Exception as e:
        print(f"❌ Error while recognizing plate: {e}")

    # Reset checking flag and record time
    is_checking = False
    last_check_time = time.time()

def start_capturing():
    """
    Main loop: read frames, detect motion in the ROI, and trigger plate capture.
    """
    global is_checking, last_check_time

    # Prime the loop with an initial frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        cap.release()
        exit()

    prev_frame = cv2.resize(prev_frame, (WIDTH, HEIGHT))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    is_checking = False
    last_check_time = 0
    COOLDOWN_AFTER_FINISH = 1  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare current frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract ROI for motion detection
        x, y, w, h = roi
        prev_roi = prev_gray[y:y + h, x:x + w]
        curr_roi = gray[y:y + h, x:x + w]

        # Detect motion via frame differencing
        diff = cv2.absdiff(prev_roi, curr_roi)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        movement_detected = any(cv2.contourArea(c) > min_area for c in contours)

        current_time = time.time()
        # If movement detected and cooldown elapsed, start capture thread
        if movement_detected and not is_checking and (current_time - last_check_time > COOLDOWN_AFTER_FINISH):
            is_checking = True
            threading.Thread(target=capture_plate).start()

        # Draw ROI rectangle and display
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Camera (16:9)", frame)

        prev_gray = gray

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Initialize camera
cap = cv2.VideoCapture(1)
WIDTH, HEIGHT = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Region of interest and motion detection params
roi = (420, 400, 400, 150)  # x, y, width, height
threshold = 25
min_area = 500

# Load any pre-approved plates and start processing
save_known_plates()
start_capturing()

# Cleanup
cap.release()
cv2.destroyAllWindows()
