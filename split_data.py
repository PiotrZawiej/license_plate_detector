import os
import shutil
import random

source_folder = r"dataset"
labels_folder = os.path.join(source_folder, "labeles")

train_folder = os.path.join(source_folder, "train")
val_folder = os.path.join(source_folder, "val")
test_folder = os.path.join(source_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

valid_ext = ('.jpg', '.jpeg', '.png')
images = [f for f in os.listdir(source_folder)
          if f.lower().endswith(valid_ext)
          and os.path.isfile(os.path.join(source_folder, f))]

random.shuffle(images)

total = len(images)
train_end = int(total * 0.6)
val_end = train_end + int(total * 0.1)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

def move_with_label(image_name, dest_folder):
    base_name, _ = os.path.splitext(image_name)
    label_name = base_name + '.txt'

    image_src = os.path.join(source_folder, image_name)
    image_dest = os.path.join(dest_folder, image_name)

    shutil.move(image_src, image_dest)

    label_src = os.path.join(labels_folder, label_name)
    label_dest = os.path.join(dest_folder, label_name)

    if os.path.exists(label_src):
        shutil.move(label_src, label_dest)
    else:
        print(f"Brak etykiety: {label_name}")

for img in train_images:
    move_with_label(img, train_folder)

for img in val_images:
    move_with_label(img, val_folder)

for img in test_images:
    move_with_label(img, test_folder)

print("end")
