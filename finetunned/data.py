import os
import shutil
import random

# ===== CONFIG =====
BASE_PATH = "VisDrone"   # root dataset folder
SEQ_PATH = os.path.join(BASE_PATH, "sequences")
ANN_PATH = os.path.join(BASE_PATH, "annotations")

OUTPUT = "dataset"

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

TRAIN_SPLIT = 0.8
# ==================

for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT}/labels/{split}", exist_ok=True)

all_data = []

# ===== LOOP THROUGH SEQUENCES =====
for seq in os.listdir(SEQ_PATH):
    seq_img_dir = os.path.join(SEQ_PATH, seq)
    ann_file = os.path.join(ANN_PATH, f"{seq}.txt")

    if not os.path.exists(ann_file):
        continue

    # Read annotations
    with open(ann_file, "r") as f:
        lines = f.readlines()

    frame_dict = {}

    for line in lines:
        data = line.strip().split(",")

        frame_id = int(data[0])
        x = float(data[2])
        y = float(data[3])
        w = float(data[4])
        h = float(data[5])

        # YOLO conversion
        x_c = (x + w/2) / IMG_WIDTH
        y_c = (y + h/2) / IMG_HEIGHT
        w /= IMG_WIDTH
        h /= IMG_HEIGHT

        class_id = 0

        if frame_id not in frame_dict:
            frame_dict[frame_id] = []

        frame_dict[frame_id].append(
            f"{class_id} {x_c} {y_c} {w} {h}"
        )

    # Save data reference
    for frame_id in frame_dict:
        img_name = f"{frame_id:07d}.jpg"
        img_path = os.path.join(seq_img_dir, img_name)

        if os.path.exists(img_path):
            all_data.append((img_path, frame_dict[frame_id]))

# ===== SPLIT =====
random.shuffle(all_data)
split_idx = int(len(all_data) * TRAIN_SPLIT)

train_data = all_data[:split_idx]
val_data = all_data[split_idx:]


def save_split(data, split):
    for i, (img_path, labels) in enumerate(data):
        new_name = f"{split}_{i:06d}.jpg"

        # copy image
        shutil.copy(img_path, f"{OUTPUT}/images/{split}/{new_name}")

        # write label
        with open(f"{OUTPUT}/labels/{split}/{new_name.replace('.jpg','.txt')}", "w") as f:
            f.write("\n".join(labels))


save_split(train_data, "train")
save_split(val_data, "val")

print("✅ Done. Dataset ready for YOLO.")