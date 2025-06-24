# License Plate Recognition

A Python project for detecting and reading vehicle license plates in images and video streams. Combines a custom-trained YOLO detector with OCR via Fast ALPR.

## Features

- **Batch evaluation**: Compute OCR accuracy and timing on a folder of test images with ground-truth annotations.
- **Live capture**: Monitor a video stream (e.g. USB camera), detect motion in a fixed ROI, extract the plate region, run OCR, and check against a trusted list.
- **Easy customization**: Swap in your own YOLO weights, adjust region-of-interest and motion-detection parameters, or plug in different OCR models.

---

## Requirements

- Python 3.8+  
- `opencv-python`  
- `ultralytics`  
- `fast-alpr`  
- `xml.etree.ElementTree` (built-in)

---
