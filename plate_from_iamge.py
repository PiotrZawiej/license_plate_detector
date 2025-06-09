def extract_plate_from_image(image, model):
    results = model(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    box = boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    cropped_plate = image[y1:y2, x1:x2]
    return cropped_plate


