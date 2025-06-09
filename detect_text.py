from plate_from_iamge import extract_plate_from_image
from ocr import ocr

from ultralytics import YOLO
import cv2

model = YOLO(r'runs\detect\train3\weights\best.pt')
image = cv2.imread(r'dataset\test\13.jpg')

plate_crop = extract_plate_from_image(image, model)

if plate_crop is not None:
    cv2.imshow("plate_crop", plate_crop)
    cv2.waitKey(0)
else:
    print("no plate")

ocr(plate_crop)
