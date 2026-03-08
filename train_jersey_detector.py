"""
Convert jerseydetection COCO dataset → YOLO format, then train YOLOv8.
Only keeps the 'jerseynumber' class (COCO id=2).

Run:
    python train_jersey_detector.py
"""

import json, os, shutil
from pathlib import Path
from ultralytics import YOLO

# ── paths ─────────────────────────────────────────────────────────────────────
COCO_DIR   = Path("/Users/pascal-maker/python-opencv/jerseydetection/train")
YOLO_DIR   = Path("/Users/pascal-maker/python-opencv/jerseydetection_yolo")
IMAGES_OUT = YOLO_DIR / "images" / "train"
LABELS_OUT = YOLO_DIR / "labels" / "train"

TARGET_COCO_ID = 2      # 'jerseynumber' in the COCO annotation file
YOLO_CLASS_ID  = 0      # remapped to 0 in YOLO format (single class)

# ── convert ───────────────────────────────────────────────────────────────────
IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LABELS_OUT.mkdir(parents=True, exist_ok=True)

with open(COCO_DIR / "_annotations.coco.json") as f:
    coco = json.load(f)

# Build lookup: image_id -> filename + dimensions
id_to_img = {img["id"]: img for img in coco["images"]}

# Group annotations by image_id, keeping only jerseynumber
from collections import defaultdict
anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["category_id"] == TARGET_COCO_ID:
        anns_by_image[ann["image_id"]].append(ann)

converted = 0
for img_id, anns in anns_by_image.items():
    img_info = id_to_img[img_id]
    fname    = img_info["file_name"]
    W, H     = float(img_info["width"]), float(img_info["height"])

    src = COCO_DIR / fname
    if not src.exists():
        print(f"  [skip] image not found: {src}")
        continue

    shutil.copy(src, IMAGES_OUT / fname)

    label_lines = []
    for ann in anns:
        x, y, w, h = [float(v) for v in ann["bbox"]]  # COCO: top-left x,y + w,h
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H
        label_lines.append(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    stem = Path(fname).stem
    with open(LABELS_OUT / f"{stem}.txt", "w") as lf:
        lf.write("\n".join(label_lines))
    converted += 1

print(f"Converted {converted} images with jerseynumber annotations.")

# ── write data.yaml ───────────────────────────────────────────────────────────
yaml_path = YOLO_DIR / "data.yaml"
yaml_path.write_text(f"""\
path: {YOLO_DIR}
train: images/train
val:   images/train   # small dataset — reuse train as val

nc: 1
names: ['jerseynumber']
""")
print(f"data.yaml written to {yaml_path}")

# ── train ─────────────────────────────────────────────────────────────────────
print("\nStarting training …")
model = YOLO("yolov8n.pt")
results = model.train(
    data=str(yaml_path),
    epochs=60,
    imgsz=640,
    batch=8,
    name="jersey_detector",
    project=str(YOLO_DIR / "runs"),
    exist_ok=True,
    patience=20,
    augment=True,
    degrees=10,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.5,
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.4,
)

best_pt = YOLO_DIR / "runs" / "jersey_detector" / "weights" / "best.pt"
print(f"\nTraining done. Best weights: {best_pt}")
