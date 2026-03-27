"""
train_traffic_yolo.py
Fine-tunes YOLOv8 on a traffic dataset for dashcam/road detection.

Supported public datasets (download separately):
  - VisDrone  : https://github.com/VisDrone/VisDrone-Dataset
  - COCO traffic subset (cars, trucks, buses, motorcycles, traffic lights)
  - UA-DETRAC : http://detrac-db.rit.albany.edu/

Dataset folder structure expected:
  data/
    images/
      train/  *.jpg
      val/    *.jpg
    labels/
      train/  *.txt   (YOLO format)
      val/    *.txt
  traffic.yaml

traffic.yaml example:
  path: ./data
  train: images/train
  val:   images/val
  names:
    0: car
    1: truck
    2: bus
    3: motorcycle
    4: bicycle
    5: traffic light
    6: person
"""

from ultralytics import YOLO

# =========================
# CONFIG
# =========================
DATA_YAML    = "traffic.yaml"   # path to your dataset YAML
BASE_WEIGHTS = "yolov8n.pt"     # start from nano (fast); use yolov8s.pt for better accuracy
OUTPUT_NAME  = "traffic_yolo_finetuned"
EPOCHS       = 3
IMG_SIZE     = 640
BATCH        = 8              # reduce to 8 if OOM on GPU

# =========================
# TRAIN
# =========================
model = YOLO(BASE_WEIGHTS)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    name=OUTPUT_NAME,
    project="runs/train",
    patience=10,             # early stopping
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    augment=True,
    degrees=5.0,             # slight rotation augment for dashcam angles
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    mosaic=1.0,
    cache=False,
    device="cpu",                # GPU 0; set to "cpu" if no GPU
    verbose=True,
    save=True,
    save_period=10,
)

print("\nTraining complete.")
print(f"Best weights: runs/train/{OUTPUT_NAME}/weights/best.pt")
print("Set YOLO_TRAFFIC_MODEL in config.py to that path.")

# =========================
# QUICK VALIDATION
# =========================
metrics = model.val()
print(f"\nmAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
