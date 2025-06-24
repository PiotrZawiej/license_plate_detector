from accuracy import accuracy_n_time
from iou import calculate_iou

accuracy, time = accuracy_n_time()
iou = calculate_iou()

print(  f"accuracy: {accuracy}\n "
        f"time: {time}\n" \
        f"iou: {iou}")