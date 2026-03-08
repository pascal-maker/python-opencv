"""
Football tracking - Detection + SAM2 Tracking + Team Assignment + Jersey OCR + Roster

Model classes for yolov8-soccer.pt:
  0: ball  1: goalkeeper  2: referee  3: soccer-player

Jersey detector (train_jersey_detector.py):
  0: jerseynumber
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO
from sports.common.team import TeamClassifier
from collections import deque

# SAM2
SAM2_CHECKPOINT = "/Users/pascal-maker/segment-anything-model-2/checkpoints/sam2_hiera_base_plus.pt"
SAM2_CONFIG     = "sam2_hiera_b+.yaml"

sys.path.insert(0, "/Users/pascal-maker/segment-anything-model-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# EasyOCR
import easyocr

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH        = "yolov8-soccer.pt"
JERSEY_MODEL_PATH = "jerseydetection_yolo/runs/jersey_detector/weights/best.pt"  # trained by train_jersey_detector.py
VIDEO_PATH        = "/Users/pascal-maker/Desktop/fcberdi.MOV"
OUTPUT_PATH       = "/Users/pascal-maker/Desktop/fcberdi_tracked.mp4"

CONF           = 0.3
PLAYER_IDS     = [1, 3]        # goalkeeper=1, soccer-player=3
CROP_STRIDE    = 5             # frames to stride when collecting crops
MAX_CROPS      = 200
SAM2_EVERY     = 3             # run SAM2 mask prediction every N frames (1 = every frame, slower)
OCR_CONF_MIN   = 0.20
MAX_FRAMES     = None          # set to None to process full video
TEAM_UPDATE_EVERY = 15         # refresh cached team ID for a tracker every N frames
OCR_EVERY      = 2             # run OCR every N frames
OCR_PHASE      = 1             # offset OCR schedule so it rarely collides with SAM2 runs

# ── Roster: fill in your players' jersey numbers → names ─────────────────────
ROSTER = {
    "1":  "De Brauwere",
    "2":  "Verhoie",
    "3":  "Reynvoet",
    "5":  "Stevens M.",
    "6":  "Vermeeren B.",
    "7":  "Ramos",
    "8":  "De Geyter",
    "9":  "Speeckaert",
    "10": "Vermeeren M.",
    "11": "van der Linden",
    "12": "Burssens",
    "13": "Reyntjens",
    "14": "Stevens T.",
    "17": "Stevens D.",
    "95": "Isahakyan",
    "96": "Delrue",
    "97": "Decoster",
    "98": "Scheire",
    "99": "Goossens",
}
VALID_NUMBERS = set(ROSTER.keys())
# ─────────────────────────────────────────────────────────────────────────────


# ── ConsecutiveValueTracker ───────────────────────────────────────────────────
class ConsecutiveValueTracker:
    """Confirms a value only after it appears n consecutive times for a tracker_id."""
    def __init__(self, n_consecutive: int = 3):
        self.n = n_consecutive
        self.history: dict[int, deque] = {}
        self.validated: dict[int, str] = {}
        self.scores: dict[int, dict[str, float]] = {}
        self.provisional: dict[int, str] = {}

    def update(self, tracker_ids: list, values: list, confidences: list[float] | None = None):
        if confidences is None:
            confidences = [0.0] * len(values)
        for tid, val, conf in zip(tracker_ids, values, confidences):
            tid = int(tid)
            val = str(val)
            conf = float(conf)
            if not val:
                continue

            # Once validated, reject weak conflicting reads to prevent flicker.
            if tid in self.validated:
                existing = self.validated[tid]
                existing_score = self.scores.get(tid, {}).get(existing, 0.0)
                new_score = self.scores.get(tid, {}).get(val, 0.0) + max(conf, 0.05)
                if val != existing and new_score < existing_score * 0.75:
                    continue

            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.n)
            self.history[tid].append(val)
            hist = self.history[tid]
            if len(hist) == self.n and len(set(hist)) == 1:
                self.validated[tid] = hist[0]

            # Accept strong single-shot OCR immediately.
            if conf >= 0.72:
                self.validated[tid] = val

            # Confidence-weighted voting fallback.
            if tid not in self.scores:
                self.scores[tid] = {}
            self.scores[tid][val] = self.scores[tid].get(val, 0.0) + max(conf, 0.05)
            best_val, best_score = max(self.scores[tid].items(), key=lambda kv: kv[1])
            self.provisional[tid] = best_val
            if best_score >= 1.2:
                self.validated[tid] = best_val

    def get_validated(self, tracker_ids: list) -> list:
        return [self.validated.get(int(tid), "") for tid in tracker_ids]

    def get_display(self, tracker_ids: list) -> list:
        """Prefer validated number; fall back to best provisional candidate."""
        out = []
        for tid in tracker_ids:
            tid = int(tid)
            out.append(self.validated.get(tid, self.provisional.get(tid, "")))
        return out


def get_sam2_masks(sam2: SAM2ImagePredictor, frame: np.ndarray,
                   detections: sv.Detections) -> sv.Detections:
    """Run SAM2 image predictor with YOLO boxes to get segmentation masks."""
    if len(detections) == 0:
        return detections
    with torch.inference_mode():
        sam2.set_image(frame)
        boxes = detections.xyxy  # (N, 4)
        masks_out, _, _ = sam2.predict(box=boxes, multimask_output=False)
    # masks_out shape: (N, 1, H, W) or (1, H, W) if single box
    if masks_out.ndim == 3:
        masks_out = masks_out[None, ...]       # (1, 1, H, W)
    masks = masks_out[:, 0, :, :].astype(bool) # (N, H, W)
    result = sv.Detections(
        xyxy=detections.xyxy,
        mask=masks,
        class_id=detections.class_id,
        confidence=detections.confidence,
        tracker_id=detections.tracker_id,
    )
    return result


def player_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    crops = []
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        crop = frame[y1:y2, x1:x2]
        crops.append(crop if crop.size > 0 else np.zeros((32, 32, 3), np.uint8))
    return crops


def number_region_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """Fallback: return torso/chest region of a player box (rows 30–60% of height)."""
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    ny1 = y1 + h * 0.30
    ny2 = y1 + h * 0.60
    return np.array([x1, ny1, x2, ny2])


def detect_jersey_box(jersey_model: YOLO | None, frame: np.ndarray,
                      player_xyxy: np.ndarray, conf: float = 0.25) -> np.ndarray | None:
    """
    Run the jersey number detector on the player crop.
    Returns the best-confidence jersey-number box in FULL-FRAME coordinates,
    or None if nothing found (caller falls back to number_region_xyxy).
    """
    if jersey_model is None:
        return None
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = map(int, np.clip(player_xyxy, [0, 0, 0, 0], [fw, fh, fw, fh]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    result = jersey_model(crop, conf=conf, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    # Pick highest-confidence detection
    best = result.boxes[int(result.boxes.conf.argmax())]
    bx1, by1, bx2, by2 = best.xyxy[0].cpu().numpy()
    # Translate back to full-frame coords
    return np.array([x1 + bx1, y1 + by1, x1 + bx2, y1 + by2])


def _upscale_for_ocr(crop_bgr: np.ndarray, min_h: int = 96) -> np.ndarray:
    """Upscale small crops to stabilize OCR."""
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return crop_bgr
    if h >= min_h:
        return crop_bgr
    scale = min_h / float(h)
    return cv2.resize(crop_bgr, (max(1, int(w * scale)), min_h), interpolation=cv2.INTER_CUBIC)


def _preprocess_for_ocr(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Return multiple preprocessed variants for EasyOCR to try."""
    crop_bgr = _upscale_for_ocr(crop_bgr)

    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(crop_bgr, -1, k)

    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=4
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    return [otsu, otsu_inv, adaptive, adaptive_inv]


def _best_ocr_candidate(results: list, min_conf: float = OCR_CONF_MIN) -> tuple[str, float]:
    """Pick longest valid digit sequence (1..99), then highest confidence."""
    candidates: list[tuple[str, float]] = []
    for item in results:
        if len(item) < 3:
            continue
        text = "".join(ch for ch in str(item[1]).strip() if ch.isdigit())
        conf = float(item[2])
        if conf < min_conf or not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if 1 <= value <= 99:
            normalized = str(value)  # normalize leading zeros (e.g. "07" -> "7")
            if normalized in VALID_NUMBERS:
                candidates.append((normalized, conf))

    if not candidates:
        return "", 0.0
    candidates.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    return candidates[0]


def _ocr_player_crop(reader: easyocr.Reader, crop_bgr: np.ndarray) -> tuple[str, float]:
    """Run OCR on a crop and keep the best confidence across preprocessing variants."""
    if crop_bgr.size == 0:
        return "", 0.0
    variants = _preprocess_for_ocr(crop_bgr)
    best_num, best_conf = "", 0.0
    for variant in variants:
        results = reader.readtext(variant, allowlist="0123456789", detail=1)
        num, conf = _best_ocr_candidate(results)
        if conf > best_conf:
            best_num, best_conf = num, conf
    return best_num, best_conf


def _ocr_player_full_box(reader: easyocr.Reader, frame: np.ndarray, xyxy: np.ndarray) -> tuple[str, float]:
    """Run OCR on full player box because number may be on front or back."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = map(int, np.clip(xyxy, [0, 0, 0, 0], [fw, fh, fw, fh]))
    crop = frame[y1:y2, x1:x2]
    return _ocr_player_crop(reader, crop)


def ocr_number_regions(reader: easyocr.Reader, frame: np.ndarray,
                       tracked: sv.Detections,
                       jersey_model: "YOLO | None" = None) -> tuple[list[str], np.ndarray, list[float]]:
    """
    OCR each player with three-stage fallback:
      1) jersey detector box  (tight, most accurate)
      2) heuristic torso region
      3) full player box
    Return:
      - numbers: list of digit strings (one per player, "" if nothing found)
      - num_boxes: (N,4) array of number-region boxes (for Mask IoU matching)
      - confidences: OCR confidence per player
    """
    fh, fw = frame.shape[:2]
    numbers, num_boxes, confidences = [], [], []
    for xyxy in tracked.xyxy:
        # Stage 1: jersey detector
        detected_box = detect_jersey_box(jersey_model, frame, xyxy)
        if detected_box is not None:
            nbox = np.clip(detected_box, [0, 0, 0, 0], [fw, fh, fw, fh])
            x1, y1, x2, y2 = map(int, nbox)
            num, conf = _ocr_player_crop(reader, frame[y1:y2, x1:x2])
        else:
            num, conf = "", 0.0

        # Stage 2: heuristic torso region
        if not num:
            nbox = np.clip(number_region_xyxy(xyxy), [0, 0, 0, 0], [fw, fh, fw, fh])
            x1, y1, x2, y2 = map(int, nbox)
            num, conf = _ocr_player_crop(reader, frame[y1:y2, x1:x2])

        # Stage 3: full player box
        if not num:
            nbox = np.clip(xyxy, [0, 0, 0, 0], [fw, fh, fw, fh])  # type: ignore[assignment]
            num, conf = _ocr_player_full_box(reader, frame, xyxy)

        numbers.append(num)
        confidences.append(conf)
        num_boxes.append(nbox)
    return numbers, np.array(num_boxes), confidences


def match_numbers_to_players_iou(
    numbers: list[str],
    num_boxes: np.ndarray,
    tracked: sv.Detections,
    frame_shape: tuple,
    iou_threshold: float = 0.1,
) -> list[tuple[int, str, int]]:
    """
    Use Mask IoU (like the basketball notebook) to match number regions
    to player masks. Returns list of (tracker_id, number, number_index) tuples.
    """
    if tracked.mask is None or len(numbers) == 0:
        # Fall back: direct 1-to-1 index match
        pairs: list[tuple[int, str, int]] = []
        tids = tracked.tracker_id if tracked.tracker_id is not None else range(len(tracked))
        for num_idx, (tid, num) in enumerate(zip(tids, numbers)):
            if num:
                pairs.append((int(tid), num, num_idx))
        return pairs

    fh, fw = frame_shape[:2]
    # Build binary masks from number-region boxes
    num_masks = np.zeros((len(num_boxes), fh, fw), dtype=bool)
    for i, box in enumerate(num_boxes):
        x1, y1, x2, y2 = map(int, np.clip(box, [0, 0, 0, 0], [fw, fh, fw, fh]))
        num_masks[i, y1:y2, x1:x2] = True

    # sv.mask_iou_batch: rows=player masks, cols=number masks → shape (P, N)
    iou = sv.mask_iou_batch(
        masks_true=tracked.mask,
        masks_detection=num_masks,
    )

    pairs: list[tuple[int, str, int]] = []
    tids = tracked.tracker_id if tracked.tracker_id is not None else list(range(len(tracked)))
    for num_idx, num in enumerate(numbers):
        if not num:
            continue
        player_idx = int(np.argmax(iou[:, num_idx]))
        if iou[player_idx, num_idx] >= iou_threshold:
            pairs.append((int(tids[player_idx]), num, num_idx))
    return pairs


# ── Step 1: collect player crops for TeamClassifier ──────────────────────────
print("Collecting player crops …")
yolo = YOLO(MODEL_PATH)

all_crops = []
for frame in sv.get_video_frames_generator(VIDEO_PATH, stride=CROP_STRIDE):
    result     = yolo(frame, conf=CONF, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[np.isin(detections.class_id, PLAYER_IDS)]
    all_crops.extend(player_crops(frame, detections))
    if len(all_crops) >= MAX_CROPS:
        break

print(f"  → {len(all_crops)} crops collected")

# ── Step 2: fit TeamClassifier ────────────────────────────────────────────────
print("Fitting team classifier …")
team_classifier = None
if len(all_crops) >= 2:
    team_classifier = TeamClassifier(device="cpu")
    team_classifier.fit(all_crops)
    print("  → done")
else:
    print("  → skipped (not enough player crops)")

# ── Step 3: load SAM2 ─────────────────────────────────────────────────────────
print("Loading SAM2 …")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2 = SAM2ImagePredictor(sam2_model)
print(f"  → SAM2 loaded on {device}")

# ── Step 4: load jersey number detector ──────────────────────────────────────
jersey_model = None
if os.path.exists(JERSEY_MODEL_PATH):
    print("Loading jersey number detector …")
    jersey_model = YOLO(JERSEY_MODEL_PATH)
    print(f"  → loaded from {JERSEY_MODEL_PATH}")
else:
    print(f"  [!] Jersey model not found at {JERSEY_MODEL_PATH}")
    print("      Run train_jersey_detector.py first. Falling back to heuristic crop.")

# ── Step 5: load EasyOCR ──────────────────────────────────────────────────────
print("Loading EasyOCR …")
ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
number_tracker = ConsecutiveValueTracker(n_consecutive=3)
print("  → done")

# ── Step 6: annotators ────────────────────────────────────────────────────────
TEAM_COLORS   = sv.ColorPalette.from_hex(["#FF6B35", "#2EC4B6"])
mask_annotator  = sv.MaskAnnotator(color=TEAM_COLORS, color_lookup=sv.ColorLookup.CLASS, opacity=0.4)
box_annotator   = sv.BoxAnnotator(color=TEAM_COLORS, color_lookup=sv.ColorLookup.CLASS)
label_annotator = sv.LabelAnnotator(
    color=TEAM_COLORS, color_lookup=sv.ColorLookup.CLASS, text_color=sv.Color.BLACK
)
ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(["#FFFFFF"]))

# ── Step 7: process video ─────────────────────────────────────────────────────
print("Processing video …")
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

# Persisted state between frames
mask_cache: dict[int, np.ndarray] = {}   # tracker_id -> last SAM2 mask (H,W bool)
team_cache: dict[int, tuple[int, int]] = {}  # tracker_id -> (team_id, last_update_frame)

with sv.VideoSink(OUTPUT_PATH, video_info) as sink:
    ocr_attempts = 0
    ocr_accepted = 0
    for idx, frame in enumerate(sv.get_video_frames_generator(VIDEO_PATH)):
        if MAX_FRAMES is not None and idx >= MAX_FRAMES:
            print(f"Reached MAX_FRAMES={MAX_FRAMES}, stopping early for debug run.")
            break
        fh, fw = frame.shape[:2]

        # ── YOLO + ByteTrack tracking ─────────────────────────────────────────
        result_track = yolo.track(frame, conf=CONF, persist=True, verbose=False)[0]
        detections   = sv.Detections.from_ultralytics(result_track)
        ball         = detections[detections.class_id == 0]
        tracked      = detections[np.isin(detections.class_id, PLAYER_IDS)]

        if len(tracked) == 0:
            sink.write_frame(frame)
            continue

        # ── SAM2 masks ────────────────────────────────────────────────────────
        if idx % SAM2_EVERY == 0:
            # Refresh masks and update cache
            tracked = get_sam2_masks(sam2, frame, tracked)
            if tracked.mask is not None and tracked.tracker_id is not None:
                for tid, mask in zip(tracked.tracker_id, tracked.mask):
                    mask_cache[int(tid)] = mask
        else:
            # Reuse persisted masks — smooth between SAM2 frames, no pixel flicker
            if tracked.tracker_id is not None:
                cached = [mask_cache.get(int(tid)) for tid in tracked.tracker_id]
                if any(m is not None for m in cached):
                    blank = np.zeros((fh, fw), dtype=bool)
                    masks = np.array([m if m is not None else blank for m in cached])
                    tracked = sv.Detections(
                        xyxy=tracked.xyxy,
                        mask=masks,
                        class_id=tracked.class_id,
                        confidence=tracked.confidence,
                        tracker_id=tracked.tracker_id,
                    )

        # ── team assignment ───────────────────────────────────────────────────
        crops = player_crops(frame, tracked)
        if team_classifier is not None and tracked.tracker_id is not None:
            team_ids = np.zeros(len(tracked), dtype=int)
            refresh_idxs = []
            refresh_crops = []
            for i, tid in enumerate(tracked.tracker_id):
                tid_i = int(tid)
                cached = team_cache.get(tid_i)
                if cached is None or (idx - cached[1]) >= TEAM_UPDATE_EVERY:
                    refresh_idxs.append(i)
                    refresh_crops.append(crops[i])
                else:
                    team_ids[i] = cached[0]

            if refresh_crops:
                refreshed = np.array(team_classifier.predict(refresh_crops), dtype=int)
                for local_i, pred in zip(refresh_idxs, refreshed):
                    tid_i = int(tracked.tracker_id[local_i])
                    team_cache[tid_i] = (int(pred), idx)
                    team_ids[local_i] = int(pred)

            tracked.class_id = team_ids
        elif team_classifier is not None:
            tracked.class_id = np.array(team_classifier.predict(crops), dtype=int)
        else:
            tracked.class_id = np.zeros(len(tracked), dtype=int)

        # ── jersey OCR every 5 frames — skip already-confirmed players ────────
        if idx % OCR_EVERY == OCR_PHASE and tracked.tracker_id is not None:
            subset = tracked
            numbers, num_boxes, confidences = ocr_number_regions(ocr_reader, frame, subset, jersey_model)
            ocr_attempts += len(confidences)
            filtered = [(i, num) for i, (num, conf) in enumerate(zip(numbers, confidences)) if num and conf >= OCR_CONF_MIN]
            ocr_accepted += len(filtered)
            if filtered:
                idxs, nums = zip(*filtered)
                sub_subset = subset[list(idxs)]
                sub_boxes  = num_boxes[list(idxs)]
                pairs = match_numbers_to_players_iou(list(nums), sub_boxes, sub_subset, frame.shape)
                if pairs:
                    p_tids = [p[0] for p in pairs]
                    p_nums = [p[1] for p in pairs]
                    pair_confs = [confidences[idxs[p[2]]] for p in pairs]
                    number_tracker.update(list(p_tids), list(p_nums), pair_confs)
                    for t, n, c in zip(p_tids, p_nums, pair_confs):
                        print(f"    OCR: tracker={t} -> #{n} (conf={c:.2f})")

        # ── build labels ──────────────────────────────────────────────────────
        tids    = tracked.tracker_id if tracked.tracker_id is not None else [None] * len(tracked)
        val_nums = number_tracker.get_display(list(tids)) if tracked.tracker_id is not None else [""] * len(tracked)

        labels = []
        for tid, team, num in zip(tids, tracked.class_id, val_nums):
            name  = ROSTER.get(num, "") if num else ""
            label = f"T{team}"
            if num:
                label += f" #{num}"
            if name:
                label += f" {name}"
            labels.append(label)

        # ── annotate ──────────────────────────────────────────────────────────
        annotated = frame.copy()
        if len(ball):
            annotated = ellipse_annotator.annotate(annotated, ball)
        if tracked.mask is not None:
            annotated = mask_annotator.annotate(annotated, tracked)
        annotated = box_annotator.annotate(annotated, tracked)
        annotated = label_annotator.annotate(annotated, tracked, labels=labels)

        sink.write_frame(annotated)

        if idx % 50 == 0:
            pct = round(idx / video_info.total_frames * 100) if video_info.total_frames else "?"
            print(f"  frame {idx}/{video_info.total_frames} ({pct}%)")
        if idx % 100 == 0:
            print(f"  ocr accepted {ocr_accepted}/{ocr_attempts} (conf>={OCR_CONF_MIN})")

print(f"\nDone! Saved to: {OUTPUT_PATH}")
