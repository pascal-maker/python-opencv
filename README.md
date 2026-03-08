# python-opencv — Football Tracking & Computer Vision

A collection of computer vision scripts built with OpenCV, YOLOv8, SAM2, and EasyOCR. The main project is a real-time football player tracking pipeline with team assignment, jersey number detection, and roster lookup.

---

## Football Tracking (`football_tracking.py`)

Full pipeline for analysing football footage:

- **Player & ball detection** — YOLOv8 trained on soccer footage
- **Multi-object tracking** — ByteTrack via Ultralytics
- **Segmentation masks** — SAM2 (Segment Anything Model 2) per player
- **Team classification** — colour-based clustering with `sports` library
- **Jersey number detection** — dedicated YOLOv8 detector trained on FCBerdi footage + EasyOCR fallback
- **Roster lookup** — maps jersey numbers to player names

### Setup

```bash
conda activate /path/to/your/env
pip install ultralytics supervision easyocr sports
pip install torch==2.4.0 torchvision==0.19.0
```

Clone SAM2 locally (required):
```bash
git clone https://github.com/facebookresearch/sam2 /path/to/segment-anything-model-2
```

### Run

```bash
python football_tracking.py
```

Output is saved to `/Users/pascal-maker/Desktop/fcberdi_tracked.mp4`.

To process a different video, update these lines in the script:
```python
VIDEO_PATH  = "/path/to/your/video.MOV"
OUTPUT_PATH = "/path/to/output.mp4"
MAX_FRAMES  = None   # or set a number for a short test run
```

---

## Jersey Number Detector (`train_jersey_detector.py`)

Trains a YOLOv8n model to detect jersey number regions on players.
Used by `football_tracking.py` to give EasyOCR a tight, clean crop instead of guessing the chest region.

### Dataset

Annotated via [Roboflow](https://roboflow.com) — COCO format export goes in `jerseydetection/`.
Classes: `people`, `ball`, `jerseynumber`, `logo`

### Train

```bash
python train_jersey_detector.py
```

Converts COCO → YOLO format, trains for 60 epochs, saves best weights to:
```
jerseydetection_yolo/runs/jersey_detector/weights/best.pt
```

Current model performance (41 training images):
- **mAP50: 0.42** — decent for a small dataset, improves with more annotations

To improve: upload more frames from your match footage to Roboflow, re-export, replace the `jerseydetection/` folder, and re-run training.

---

## Other Scripts

| Script | Description |
|---|---|
| `regiondetection.py` | Count objects crossing user-defined regions |
| `segmentit.py` | Instance segmentation on video |
| `voetbal.py` | Early football detection prototype |
| `berdiboys.py` | Player detection on FCBerdi footage |
| `motion-detection-security-camera.py` | Motion detection for security use |
| `boxingvideo.py` | Pose estimation on boxing footage |
| `inference.py` / `videoinference.py` | General YOLO inference helpers |

---

## Models

| File | Description |
|---|---|
| `yolov8-soccer.pt` | Main player/ball/referee detector |
| `yolov8n-football.pt` | Lightweight football detector |
| `best.pt` | General purpose fine-tuned model |
| `jerseydetection_yolo/runs/.../best.pt` | Jersey number region detector |

---

## Environment Notes

- Python 3.10, PyTorch 2.4.0, Apple M2 Max (MPS)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for SAM2 (already set in script)
- Conda env: `/Users/pascal-maker/pascalworkspace/env`
