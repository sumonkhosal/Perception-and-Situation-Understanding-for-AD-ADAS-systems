# ============================================================
# VS CODE / LOCAL PYTHON VERSION (30 FPS PRESET) - WINDOWS SAFE
# READY-TO-PASTE
#
# ✅ UPDATE #1 (Grid vs Depth mismatch FIX):
#   - Depth thresholds (p30/p60) computed on ROAD REGION ONLY
#   - BBox depth sampled from BOTTOM "feet band"
#
# ✅ UPDATE #2 (Transparent HUD):
#   - HUD background is transparent black so video stays visible
#
# ✅ UPDATE #3 (LANE DETECTION ADDED):
#   - Bird’s-eye lane detection using HOUGH (line proposals) + SLIDING WINDOW fit
#   - Draws lane on PROC frame + BEV window + BEV output video
#
# Notes:
# - MP4 "top-left only" bug on Windows: always write AVI (MJPG) correctly
# - Optional MP4 writer best-effort
# - YOLO stride warning fixed: PROC_H multiple of 32
# - Output filenames include Date+Time (RUN_TS)
# - Auto depth invert is computed, then SMOOTHED + LOCKED after warmup frames
# ============================================================

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------------------------------------------------------
# USER CONFIG
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "output"
VIDEO_PATH = ROOT_DIR / "videos" / "Prank_Video.mp4"

DEPTH_ANYTHING_REPO = ROOT_DIR / "Depth-Anything-V2"
DEPTH_CKPT_PATH = ROOT_DIR / "models" / "depth_anything_v2_vits.pth"
SIGN_WEIGHTS = ROOT_DIR / "models" / "traffic_sign_yolov8.pt"

# ------------------------------------------------------------
# PERF PRESET (IMPORTANT: multiple of 32 for YOLO)
# ------------------------------------------------------------
PROC_W = 960
PROC_H = 544  # multiple of 32 -> avoids YOLO stride warning
DEPTH_INPUT_SIZE = 256

COCO_EVERY_N = 3
SIGN_EVERY_N = 12

SHOW_LIVE = True
MAX_FPS_CAP = 30

COCO_CONF = 0.40
COCO_IOU  = 0.50
SIGN_CONF = 0.35
SIGN_IOU  = 0.50

# Default fallback (until auto-invert locks)
DEPTH_INVERT_DEFAULT = False

# Auto-invert smoothing + lock
INVERT_WARMUP_FRAMES = 60   # measure invert for ~2 sec at 30fps
INVERT_EMA = 0.90           # smooth flip/noise
INVERT_LOCK = True          # lock after warmup

# Lane detection (HOUGH + Sliding Window)
LANE_ENABLE = True
LANE_EVERY_N = 2            # run lane pipeline every N frames
LANE_ALPHA = 0.55           # overlay strength
LANE_DEBUG_DRAW_WINDOWS = False

# ------------------------------------------------------------
# Safety helpers
# ------------------------------------------------------------
def ensure_exists(path, what="file/dir"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {what}: {path}")
    
def make_writer_any(path_base_no_ext, fps, size_wh):
    """
    Windows-safe: always writes AVI (MJPG) correctly.
    Also tries MP4 best-effort (optional).
    """
    path_base_no_ext = Path(path_base_no_ext)
    path_base_no_ext.parent.mkdir(parents=True, exist_ok=True)

    W, H = size_wh

    avi_path = path_base_no_ext.with_suffix(".avi")
    avi_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    avi_writer = cv2.VideoWriter(str(avi_path), avi_fourcc, fps, (W, H))
    if not avi_writer.isOpened():
        raise RuntimeError(f"AVI(MJPG) writer failed: {avi_path}")
    print(f"[OK] AVI writer: {avi_path}  size={W}x{H} fps={fps:.2f}")

    mp4_path = path_base_no_ext.with_suffix(".mp4")
    mp4_writer = None
    for code in ["mp4v", "avc1", "H264"]:
        fourcc = cv2.VideoWriter_fourcc(*code)
        w = cv2.VideoWriter(str(mp4_path), fourcc, fps, (W, H))
        if w.isOpened():
            print(f"[OK] MP4 writer: {mp4_path} codec={code} size={W}x{H} fps={fps:.2f}")
            mp4_writer = w
            break

    return avi_writer, mp4_writer, str(avi_path), (str(mp4_path) if mp4_writer else None)

def load_depthanything_weights(model, ckpt_path, device):
    """
    Handles common checkpoint formats:
    - raw state_dict
    - {"state_dict": ...}
    - {"model": ...}
    Also strips 'module.' prefix if present.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[len("module."):]] = v
            else:
                new_sd[k] = v
        sd = new_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[DepthAnything] load_state_dict(strict=False)")
    if missing:
        print("  missing keys:", len(missing))
    if unexpected:
        print("  unexpected keys:", len(unexpected))

# ------------------------------------------------------------
# Import DepthAnythingV2
# ------------------------------------------------------------
ensure_exists(DEPTH_ANYTHING_REPO, "directory")
ensure_exists(DEPTH_CKPT_PATH, "file")
ensure_exists(SIGN_WEIGHTS, "file")

sys.path.insert(0, str(DEPTH_ANYTHING_REPO))
from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

# ==========================================================
# Transparent overlay helpers
# ==========================================================
def draw_transparent_rect(img, x1, y1, x2, y2, alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def overlay_nonzero(base_bgr, overlay_bgr, alpha=0.6):
    """Alpha-blend only where overlay has content (nonzero pixels)."""
    if overlay_bgr is None:
        return base_bgr
    mask = (overlay_bgr[..., 0] > 0) | (overlay_bgr[..., 1] > 0) | (overlay_bgr[..., 2] > 0)
    if not mask.any():
        return base_bgr
    out = base_bgr.copy()
    # Blend on mask
    out[mask] = (alpha * overlay_bgr[mask] + (1 - alpha) * base_bgr[mask]).astype(np.uint8)
    return out

def auto_invert_depth_for_people(ped_depth_pairs, default_invert=True):
    """
    Auto-detect whether higher depth value means closer or farther,
    using correlation between bbox bottom y (yb) and depth value.
    """
    pts = [(yb, d) for (yb, d) in ped_depth_pairs if d is not None]
    if len(pts) < 12:
        return default_invert

    y = np.array([p[0] for p in pts], dtype=np.float32)
    d = np.array([p[1] for p in pts], dtype=np.float32)

    y = (y - y.mean()) / (y.std() + 1e-6)
    d = (d - d.mean()) / (d.std() + 1e-6)

    corr = float((y * d).mean())
    # If corr > 0: as y increases (closer), d increases => invert=True means "bigger=closer"
    return (corr > 0)

# ==========================================================
# Core helpers
# ==========================================================
def segment_road_region(frame_bgr: np.ndarray):
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    Ss = hsv[..., 1]
    Vv = hsv[..., 2]

    roi_start_y = int(0.45 * H)

    center_x = W // 2
    center_y = H - int(0.03 * H)
    patch_half_w = int(0.10 * W)
    patch_half_h = int(0.05 * H)

    x1 = max(0, center_x - patch_half_w)
    x2 = min(W - 1, center_x + patch_half_w)
    y1 = max(roi_start_y, center_y - patch_half_h)
    y2 = min(H - 1, center_y + patch_half_h)

    patchS = Ss[y1:y2 + 1, x1:x2 + 1]
    patchV = Vv[y1:y2 + 1, x1:x2 + 1]
    total_score = patchS + np.abs(patchV - 0.5)

    py, px = np.unravel_index(np.argmin(total_score), total_score.shape)
    seed_x = x1 + px
    seed_y = y1 + py
    seedS = float(Ss[seed_y, seed_x])
    seedV = float(Vv[seed_y, seed_x])

    mask = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=np.uint8)

    max_dist = 0.18
    q = [(seed_y, seed_x)]
    visited[seed_y, seed_x] = 1
    mask[seed_y, seed_x] = 1

    head = 0
    while head < len(q):
        y, x = q[head]
        head += 1
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if yy < roi_start_y or yy >= H or xx < 0 or xx >= W:
                continue
            if visited[yy, xx]:
                continue
            visited[yy, xx] = 1
            d = np.sqrt((Ss[yy, xx] - seedS) ** 2 + (Vv[yy, xx] - seedV) ** 2)
            if d < max_dist:
                mask[yy, xx] = 1
                q.append((yy, xx))

    kernel = np.ones((3, 3), np.uint8)
    conv = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mask_smooth = (conv >= 5).astype(np.uint8)
    mask_smooth[:roi_start_y, :] = 0

    return (mask_smooth * 255).astype(np.uint8), mask_smooth.astype(bool)


def dedup_tracks_by_iou(tracks, iou_thr=0.65):
    """
    Removes overlapping tracks (keeps the one with fewer misses / older preference).
    Works even without confidences.
    """
    if not tracks:
        return tracks

    # Prefer tracks that are not missed (and optionally older/higher hits if you add that later)
    tracks_sorted = sorted(tracks, key=lambda t: (t.missed))
    kept = []
    for tr in tracks_sorted:
        drop = False
        for k in kept:
            if iou_xyxy(tr.bbox, k.bbox) >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append(tr)
    return kept

def dedup_detections_xyxy(dets_xyxy, iou_thr=0.70):
    """
    Simple IoU-based deduplication (NMS-like) without confidences.
    Keeps the largest boxes first and suppresses overlapping ones.
    """
    if not dets_xyxy:
        return dets_xyxy

    # Sort by area (largest first) so we keep the main box
    dets = sorted(
        dets_xyxy,
        key=lambda b: max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1])),
        reverse=True
    )

    kept = []
    for b in dets:
        keep = True
        for k in kept:
            if iou_xyxy(b, k) >= iou_thr:
                keep = False
                break
        if keep:
            kept.append(b)
    return kept



def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def majority(seq):
    if not seq:
        return "UNK"
    d = {}
    for s in seq:
        d[s] = d.get(s, 0) + 1
    return max(d, key=d.get)

def traffic_light_state_from_bbox(frame_bgr, bbox_xyxy):
    """
    Estimate traffic light state (RED/YELLOW/GREEN) from pixels inside bbox.
    Works best for vertical lights. Returns "RED"/"YELLOW"/"GREEN"/"UNKNOWN".
    """
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(W - 1, int(round(x1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y1 = max(0, min(H - 1, int(round(y1))))
    y2 = max(0, min(H - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN"

    crop = frame_bgr[y1:y2+1, x1:x2+1]
    if crop.shape[0] < 12 or crop.shape[1] < 8:
        return "UNKNOWN"

    # Focus on center (cuts background)
    ch, cw = crop.shape[:2]
    cx1 = int(0.15 * cw); cx2 = int(0.85 * cw)
    cy1 = int(0.05 * ch); cy2 = int(0.95 * ch)
    crop = crop[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < 12 or cw < 8:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # HSV masks (tuned for bright signal lights)
    # Red wraps around hue, so we use two ranges
    red1 = cv2.inRange(hsv, (0, 80, 130), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 130), (179, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, (15, 70, 140), (35, 255, 255))
    green  = cv2.inRange(hsv, (40, 70, 120), (90, 255, 255))

    # Split into 3 vertical sections (top/mid/bottom) -> typical vertical traffic lights
    h3 = ch // 3
    if h3 < 4:
        return "UNKNOWN"

    sections = [
        ("RED",    red[0:h3, :]),
        ("YELLOW", yellow[h3:2*h3, :]),
        ("GREEN",  green[2*h3:3*h3, :]),
    ]

    scores = {}
    for name, m in sections:
        scores[name] = float(np.mean(m > 0))  # fraction of pixels matching

    # pick best
    best = max(scores, key=scores.get)
    best_score = scores[best]

    # require minimum evidence; otherwise unknown
    if best_score < 0.02:   # 2% pixels
        return "UNKNOWN"

    return best



def pretty_sign_label(lab: str):
    s = str(lab).lower()
    if ("speed" in s) or ("limit" in s) or ("kmh" in s):
        digits = "".join([c for c in s if c.isdigit()])
        return f"SPEED {digits} km/h" if digits else "SPEED LIMIT"
    if "school" in s:
        return "SCHOOL AHEAD"
    if "stop" in s:
        return "STOP"
    if "yield" in s or "give_way" in s or "giveway" in s:
        return "YIELD"
    if "no_entry" in s or "noentry" in s:
        return "NO ENTRY"
    if "cross" in s and "ped" in s:
        return "PEDESTRIAN CROSSING"
    return str(lab)

def infer_ped_intent(side_hist, x_hist, y_hist, min_frames=10):
    if len(side_hist) < min_frames:
        return "UNKNOWN"
    recent_sides = list(side_hist)[-min_frames:]
    recent_x = list(x_hist)[-min_frames:]
    recent_y = list(y_hist)[-min_frames:]
    dx = recent_x[-1] - recent_x[0]
    dy = recent_y[-1] - recent_y[0]
    adx, ady = abs(dx), abs(dy)
    if adx > 25 and adx > 1.5 * ady:
        return "CROSS_L2R" if dx > 0 else "CROSS_R2L"
    maj = majority(recent_sides)
    stable_frac = sum(s == maj for s in recent_sides) / float(len(recent_sides))
    if ady > 25 and ady > 1.5 * adx and stable_frac >= 0.7:
        return "STRAIGHT"
    return "UNKNOWN"

def classify_side_grid(xc: float, yb: float, W: int, H: int, y_split: int, van_x: int, base_left_x: int, base_right_x: int):
    left_border = 0.40 * W
    right_border = 0.60 * W
    if yb <= y_split:
        if xc < left_border:
            return "LEFT"
        elif xc > right_border:
            return "RIGHT"
        else:
            return "FRONT"
    else:
        t = (yb - y_split) / max(1, (H - y_split))
        x_left = van_x + t * (base_left_x - van_x)
        x_right = van_x + t * (base_right_x - van_x)
        if xc < x_left:
            return "LEFT"
        elif xc > x_right:
            return "RIGHT"
        else:
            return "FRONT"

def classify_distance_far_mid_close(yb: float, H: int):
    d1 = 0.50 * H
    d2 = 0.80 * H
    if yb < d1:
        return "FAR"
    elif yb < d2:
        return "MID"
    else:
        return "CLOSE"

def infer_approach_intent(depth_hist, invert=False, min_n=10, thr=0.03):
    vals = [v for v in depth_hist if v is not None]
    if len(vals) < min_n:
        return "UNKNOWN"
    vals = np.array(vals[-min_n:], dtype=np.float32)
    v0 = float(np.median(vals))
    if abs(v0) < 1e-6:
        v0 = 1.0
    valsn = vals / v0
    slope = float(valsn[-1] - valsn[0]) / float(min_n - 1)

    if not invert:
        if slope < -thr: return "APPROACHING"
        if slope > thr:  return "MOVING_AWAY"
        return "STABLE"
    else:
        if slope > thr:  return "APPROACHING"
        if slope < -thr: return "MOVING_AWAY"
        return "STABLE"

def depth_to_colormap_stable(depth_raw, state, ema=0.92):
    d = depth_raw.astype(np.float32)
    lo = np.percentile(d, 5)
    hi = np.percentile(d, 95)
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    if state["lo"] is None:
        state["lo"], state["hi"] = lo, hi
    else:
        state["lo"] = ema * state["lo"] + (1 - ema) * lo
        state["hi"] = ema * state["hi"] + (1 - ema) * hi
    lo_s, hi_s = state["lo"], state["hi"]
    dn = np.clip((d - lo_s) / (hi_s - lo_s + 1e-6), 0.0, 1.0)
    img8 = (dn * 255).astype(np.uint8)
    return cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)

# ✅ UPDATE #1A: feet-band bbox depth (more stable than full bbox median)
def robust_bbox_depth(depth_raw: np.ndarray, bbox_xyxy, H: int, W: int):
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(W - 1, int(round(x1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y1 = max(0, min(H - 1, int(round(y1))))
    y2 = max(0, min(H - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None

    h = y2 - y1 + 1
    band_y1 = y2 - int(0.35 * h)  # bottom 35%
    band_y1 = max(y1, band_y1)

    patch = depth_raw[band_y1:y2 + 1, x1:x2 + 1].astype(np.float32)
    if patch.size < 50:
        return None
    return float(np.median(patch))

# ✅ UPDATE #1B: thresholds computed on ROAD region only
def update_depth_thresholds(depth_raw: np.ndarray, road_mask_bool: np.ndarray, state: dict, ema=0.92):
    d = depth_raw.astype(np.float32)
    if road_mask_bool is not None and road_mask_bool.any():
        vals = d[road_mask_bool]
    else:
        vals = d.reshape(-1)

    p30 = float(np.percentile(vals, 30))
    p60 = float(np.percentile(vals, 60))

    if state["p30"] is None:
        state["p30"], state["p60"] = p30, p60
    else:
        state["p30"] = ema * state["p30"] + (1 - ema) * p30
        state["p60"] = ema * state["p60"] + (1 - ema) * p60

    return state["p30"], state["p60"]

def depth_label_from_value(dval: float, p30: float, p60: float, invert=False):
    if dval is None:
        return "UNK"
    if not invert:
        if dval <= p30: return "CLOSE"
        if dval <= p60: return "MID"
        return "FAR"
    else:
        if dval >= p60: return "CLOSE"
        if dval >= p30: return "MID"
        return "FAR"

def update_grid_depth_mismatch(false_counts, grid_label, depth_label):
    if depth_label == "UNK":
        return
    if grid_label == "CLOSE" and depth_label != "CLOSE":
        false_counts["false_close"][depth_label] += 1
    elif grid_label == "MID" and depth_label != "MID":
        false_counts["false_mid"][depth_label] += 1
    elif grid_label == "FAR" and depth_label != "FAR":
        false_counts["false_far"][depth_label] += 1

# ==========================================================
# Trackers (IOU + Kalman)
# ==========================================================
class PersonTrack:
    def __init__(self, display_id, init_bbox, dt=1.0):
        self.display_id = display_id
        self.side_hist = deque(maxlen=30)
        self.x_hist = deque(maxlen=30)
        self.y_hist = deque(maxlen=30)
        self.last_intent = "UNKNOWN"
        self.missed = 0

        x1, y1, x2, y2 = init_bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        self.w = max(1.0, (x2 - x1))
        self.h = max(1.0, (y2 - y1))
        self.bbox = init_bbox

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float32)
    def predict(self):
        pred = self.kf.predict()
        cx, cy = float(pred[0, 0]), float(pred[1, 0])
        x1 = cx - 0.5 * self.w
        y1 = cy - 0.5 * self.h
        x2 = cx + 0.5 * self.w
        y2 = cy + 0.5 * self.h
        self.bbox = (x1, y1, x2, y2)


    def update(self, det_bbox):
        x1, y1, x2, y2 = det_bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))
        w = max(1.0, (x2 - x1))
        h = max(1.0, (y2 - y1))
        self.w = 0.8 * self.w + 0.2 * w
        self.h = 0.8 * self.h + 0.2 * h
        self.bbox = det_bbox
        self.missed = 0

class PersonTracker:
    def __init__(self, max_tracks=7, iou_thr=0.25, max_missed=15):
        self.max_tracks = max_tracks
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks = []
        self._next_display_id = 1

    def update(self, detections_xyxy):
        for tr in self.tracks:
            tr.predict()

        unmatched_dets = set(range(len(detections_xyxy)))
        unmatched_trs = set(range(len(self.tracks)))

        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections_xyxy):
                pairs.append((iou_xyxy(tr.bbox, det), ti, di))
        pairs.sort(reverse=True, key=lambda t: t[0])

        for iou_val, ti, di in pairs:
            if iou_val < self.iou_thr:
                break
            if ti in unmatched_trs and di in unmatched_dets:
                unmatched_trs.remove(ti)
                unmatched_dets.remove(di)
                self.tracks[ti].update(detections_xyxy[di])

        for ti in list(unmatched_trs):
            self.tracks[ti].missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for di in list(unmatched_dets):
            if len(self.tracks) >= self.max_tracks:
                break
            self.tracks.append(PersonTrack(self._next_display_id, detections_xyxy[di]))
            self._next_display_id += 1

        return self.tracks

class GenericTrack:
    def __init__(self, display_id, init_bbox, init_label="UNK", dt=1.0):
        self.display_id = display_id
        self.label_hist = deque(maxlen=20)
        self.label_hist.append(init_label)
        self.label = init_label
        self.depth_hist = deque(maxlen=25)
        self.intent = "UNKNOWN"
        self.missed = 0

        self.w = max(1.0, (init_bbox[2] - init_bbox[0]))
        self.h = max(1.0, (init_bbox[3] - init_bbox[1]))
        self.bbox = init_bbox

        x1, y1, x2, y2 = init_bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        cx, cy = float(pred[0, 0]), float(pred[1, 0])
        x1 = cx - 0.5 * self.w
        y1 = cy - 0.5 * self.h
        x2 = cx + 0.5 * self.w
        y2 = cy + 0.5 * self.h
        self.bbox = (x1, y1, x2, y2)

    def update(self, det_bbox, det_label=None):
        x1, y1, x2, y2 = det_bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))

        w = max(1.0, (x2 - x1))
        h = max(1.0, (y2 - y1))
        self.w = 0.8 * self.w + 0.2 * w
        self.h = 0.8 * self.h + 0.2 * h

        self.bbox = det_bbox
        self.missed = 0

        if det_label is not None:
            self.label_hist.append(det_label)
            self.label = majority(list(self.label_hist))

class GenericTracker:
    def __init__(self, max_tracks=12, iou_thr=0.25, max_missed=25):
        self.max_tracks = max_tracks
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks = []
        self._next_display_id = 1

    def update(self, detections_xyxy, det_labels=None):
        if det_labels is None:
            det_labels = ["UNK"] * len(detections_xyxy)

        for tr in self.tracks:
            tr.predict()

        unmatched_dets = set(range(len(detections_xyxy)))
        unmatched_trs = set(range(len(self.tracks)))

        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections_xyxy):
                pairs.append((iou_xyxy(tr.bbox, det), ti, di))
        pairs.sort(reverse=True, key=lambda t: t[0])

        for iou_val, ti, di in pairs:
            if iou_val < self.iou_thr:
                break
            if ti in unmatched_trs and di in unmatched_dets:
                unmatched_trs.remove(ti)
                unmatched_dets.remove(di)
                self.tracks[ti].update(detections_xyxy[di], det_labels[di])

        for ti in list(unmatched_trs):
            self.tracks[ti].missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for di in list(unmatched_dets):
            if len(self.tracks) >= self.max_tracks:
                break
            self.tracks.append(GenericTrack(self._next_display_id, detections_xyxy[di], init_label=det_labels[di]))
            self._next_display_id += 1

        return self.tracks

# ==========================================================
# BEV helpers
# ==========================================================
def get_bev_matrices(W, H, bevW=640, bevH=720):
    SRC = np.float32([
        [0.43 * W, 0.62 * H],
        [0.57 * W, 0.62 * H],
        [0.93 * W, 0.98 * H],
        [0.07 * W, 0.98 * H],
    ])
    DST = np.float32([
        [0.20 * bevW, 0.00 * bevH],
        [0.80 * bevW, 0.00 * bevH],
        [0.80 * bevW, 1.00 * bevH],
        [0.20 * bevW, 1.00 * bevH],
    ])
    H_img2bev = cv2.getPerspectiveTransform(SRC, DST)
    H_bev2img = cv2.getPerspectiveTransform(DST, SRC)
    return H_img2bev, H_bev2img, (bevW, bevH)

def warp_to_bev(frame_bgr, H_img2bev, bev_size):
    bevW, bevH = bev_size
    return cv2.warpPerspective(frame_bgr, H_img2bev, (bevW, bevH), flags=cv2.INTER_LINEAR)

# ==========================================================
# Lane detection: HOUGH (proposal) + SLIDING WINDOW (fit)
# ==========================================================
def _lane_edges_bev(bev_bgr):
    gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Light normalization to help in shadows
    blur = cv2.equalizeHist(blur)

    edges = cv2.Canny(blur, 60, 160)

    # Keep mostly center area in BEV
    H, W = edges.shape
    roi = np.zeros_like(edges)
    poly = np.array([[
        (int(0.10 * W), H-1),
        (int(0.90 * W), H-1),
        (int(0.70 * W), int(0.08 * H)),
        (int(0.30 * W), int(0.08 * H)),
    ]], dtype=np.int32)
    cv2.fillPoly(roi, poly, 255)
    edges = cv2.bitwise_and(edges, roi)

    return edges

def lane_hough_sliding(bev_bgr, draw_windows=False):
    """
    Returns:
      lane_overlay_bev (BGR) : drawn lane polygon/lines in BEV
      lane_binary (uint8)    : lane pixels used for sliding window
      ok (bool)
    """
    H, W = bev_bgr.shape[:2]
    edges = _lane_edges_bev(bev_bgr)

    # HOUGH -> draw thick lines into a binary mask (this becomes the "lane pixels")
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=55,
        minLineLength=35,
        maxLineGap=140
    )

    lane_mask = np.zeros((H, W), dtype=np.uint8)
    if lines is not None:
        for l in lines[:, 0]:
            x1, y1, x2, y2 = map(int, l)

            # Filter out near-horizontal segments in BEV
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) < 20:
                continue

            # Draw thick
            cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 8)

    # Strengthen + connect
    kernel = np.ones((5, 5), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    binary = (lane_mask > 0).astype(np.uint8)

    # Sliding window fit
    histogram = np.sum(binary[H//2:, :], axis=0)
    midpoint = W // 2
    leftx_base = int(np.argmax(histogram[:midpoint])) if histogram[:midpoint].max() > 0 else None
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint) if histogram[midpoint:].max() > 0 else None

    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0], dtype=np.int32)
    nonzerox = np.array(nonzero[1], dtype=np.int32)

    nwindows = 9
    window_height = H // nwindows
    margin = 70
    minpix = 45

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    overlay = bev_bgr.copy()

    for window in range(nwindows):
        win_y_low = H - (window + 1) * window_height
        win_y_high = H - window * window_height

        if leftx_current is not None:
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            good_left = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            left_lane_inds.append(good_left)

            if draw_windows:
                cv2.rectangle(overlay, (max(0, win_xleft_low), win_y_low), (min(W-1, win_xleft_high), win_y_high), (255, 0, 0), 2)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))

        if rightx_current is not None:
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_right = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]
            right_lane_inds.append(good_right)

            if draw_windows:
                cv2.rectangle(overlay, (max(0, win_xright_low), win_y_low), (min(W-1, win_xright_high), win_y_high), (0, 0, 255), 2)

            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=np.int64)
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=np.int64)

    # Extract lane pixels
    leftx = nonzerox[left_lane_inds] if left_lane_inds.size else np.array([])
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.size else np.array([])
    rightx = nonzerox[right_lane_inds] if right_lane_inds.size else np.array([])
    righty = nonzeroy[right_lane_inds] if right_lane_inds.size else np.array([])

    ok = (leftx.size > 200) and (rightx.size > 200)

    lane_overlay = np.zeros_like(bev_bgr)

    if ok:
        left_fit = np.polyfit(lefty.astype(np.float32), leftx.astype(np.float32), 2)
        right_fit = np.polyfit(righty.astype(np.float32), rightx.astype(np.float32), 2)

        ploty = np.linspace(0, H - 1, H).astype(np.float32)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_pts = np.stack([left_fitx, ploty], axis=1).astype(np.int32)
        right_pts = np.stack([right_fitx, ploty], axis=1).astype(np.int32)

        # Clip to image bounds
        left_pts[:, 0] = np.clip(left_pts[:, 0], 0, W-1)
        right_pts[:, 0] = np.clip(right_pts[:, 0], 0, W-1)

        # Fill polygon between lanes
        poly_pts = np.vstack([left_pts, right_pts[::-1]])
        cv2.fillPoly(lane_overlay, [poly_pts], (0, 255, 255))  # lane area

        # Draw centerline
        centerx = ((left_fitx + right_fitx) * 0.5).astype(np.int32)
        center_pts = np.stack([np.clip(centerx, 0, W-1), ploty.astype(np.int32)], axis=1)
        cv2.polylines(lane_overlay, [left_pts], isClosed=False, color=(255, 0, 0), thickness=3)
        cv2.polylines(lane_overlay, [right_pts], isClosed=False, color=(0, 0, 255), thickness=3)
        cv2.polylines(lane_overlay, [center_pts.astype(np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

    # Optional: show hough edges overlay in BEV for debugging
    if lines is not None:
        dbg = overlay.copy()
    else:
        dbg = bev_bgr.copy()

    # Return BEV overlay to be warped back to PROC
    return lane_overlay, (binary * 255).astype(np.uint8), ok

def warp_bev_overlay_to_proc(overlay_bev_bgr, H_bev2img, proc_size_wh):
    procW, procH = proc_size_wh
    return cv2.warpPerspective(overlay_bev_bgr, H_bev2img, (procW, procH), flags=cv2.INTER_LINEAR)

# ==========================================================
# HUD FULL (transparent)
# ==========================================================
def draw_hud_full(out, ped_tracks, sign_tracks, veh_tracks,
                  false_counts, total_compared,
                  dbg_depth_valid, dbg_person_dets, inv_flag, lane_ok):
    H, W = out.shape[:2]
    hud_h = 310

    draw_transparent_rect(out, 8, 8, W - 8, 8 + hud_h, alpha=0.45)

    fc = sum(false_counts["false_close"].values())
    fm = sum(false_counts["false_mid"].values())
    ff = sum(false_counts["false_far"].values())

    cv2.putText(out, f"Compared={total_compared} | False CLOSE={fc} MID={fm} FAR={ff}",
                (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(out, f"DBG: person_dets={dbg_person_dets} | depth_valid={dbg_depth_valid} | inv={int(inv_flag)} | lane_ok={int(lane_ok)}",
                (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

    y = 105
    cv2.putText(out, "PEDESTRIANS:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    y += 25
    for tr in ped_tracks[:7]:
        side = majority(list(tr.side_hist)[-10:])
        line = f"Person {tr.display_id}: intent={tr.last_intent} side={side} missed={tr.missed}"
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        y += 22

    y += 10
    cv2.putText(out, "SIGNS:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    y += 25
    for tr in sign_tracks[:6]:
        line = f"Sign {tr.display_id}: {pretty_sign_label(tr.label)} missed={tr.missed}"
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        y += 22

    y += 10
    cv2.putText(out, "VEHICLES:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    y += 25
    for tr in veh_tracks[:6]:
        line = f"{tr.label} {tr.display_id}: {tr.intent} missed={tr.missed}"
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        y += 22

    return out

# ==========================================================
# MAIN (FIXED: coco_boxes always defined + safe traffic lights)
# ==========================================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- models ---
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    }
    depth_model = DepthAnythingV2(**model_configs["vits"]).to(DEVICE).eval()
    load_depthanything_weights(depth_model, DEPTH_CKPT_PATH, DEVICE)

    coco_model = YOLO("yolov8n.pt")
    sign_model = YOLO(str(SIGN_WEIGHTS))
    

    # --- video ---
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("Could not read first frame.")
    H0, W0 = first.shape[:2]

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 0:
        fps_in = 25.0
    print(f"Input video (true): {W0}x{H0}, fps={fps_in:.2f}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- writers ---
    writer_avi, writer_mp4, out_avi, out_mp4 = make_writer_any(
    OUT_DIR / f"output_annotated_30fps_{RUN_TS}", fps_in, (W0, H0)
    )
    writer_rgb_avi, writer_rgb_mp4, out_rgb_avi, out_rgb_mp4 = make_writer_any(
        OUT_DIR / f"output_rgb_depth_30fps_{RUN_TS}", fps_in, (W0 * 2, H0)
    )

    # BEV matrices (KEEP BOTH!)
    H_img2bev, H_bev2img, bev_size = get_bev_matrices(PROC_W, PROC_H, bevW=640, bevH=720)
    bevW, bevH = bev_size
    writer_bev_avi, writer_bev_mp4, out_bev_avi, out_bev_mp4 = make_writer_any(
        OUT_DIR / f"output_bev_30fps_{RUN_TS}", fps_in, (bevW, bevH)
    )
    # --- trackers ---
    person_tracker = PersonTracker(max_tracks=7, iou_thr=0.25, max_missed=15)
    sign_tracker = GenericTracker(max_tracks=12, iou_thr=0.25, max_missed=25)
    vehicle_tracker = GenericTracker(max_tracks=12, iou_thr=0.25, max_missed=25)
    vehicle_classes = {"car", "truck", "bus", "motorcycle"}

    # --- grid params ---
    y_split = int(0.45 * PROC_H)
    van_x = PROC_W // 2
    base_left_x = int(0.05 * PROC_W)
    base_right_x = int(0.95 * PROC_W)

    # --- depth state ---
    depth_vis_state = {"lo": None, "hi": None}
    depth_thr_state = {"p30": None, "p60": None}

    false_counts = {
        "false_close": {"FAR": 0, "MID": 0, "CLOSE": 0},
        "false_mid":   {"FAR": 0, "MID": 0, "CLOSE": 0},
        "false_far":   {"FAR": 0, "MID": 0, "CLOSE": 0},
    }
    total_compared = 0

    # ✅ cache detections from last run (THIS is what avoids coco_boxes undefined)
    last_coco_boxes, last_coco_labels = [], []
    last_sign_dets, last_sign_labels = [], []

    frame_idx = 0
    t_last = time.time()

    # auto-invert state (smoothed + optional lock)
    inv_smooth = 1.0 if DEPTH_INVERT_DEFAULT else 0.0
    inv_locked = False

    # lane state
    last_lane_overlay_bev = np.zeros((bevH, bevW, 3), dtype=np.uint8)
    last_lane_ok = False

    if SHOW_LIVE:
        cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
        cv2.namedWindow("RGB|Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)

    while True:
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            break
        frame_idx += 1

        frame = cv2.resize(frame0, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)

        # ✅ IMPORTANT: coco_boxes always exists (use last cached detections if YOLO not run)
        coco_boxes, coco_labels = last_coco_boxes, last_coco_labels

        # road mask first (we need it for road-only depth thresholds)
        _, road_mask = segment_road_region(frame)

        # depth
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_raw = depth_model.infer_image(frame_rgb, input_size=DEPTH_INPUT_SIZE)

        # ROAD-only thresholds
        p30, p60 = update_depth_thresholds(depth_raw, road_mask, depth_thr_state, ema=0.92)

        depth_color = depth_to_colormap_stable(depth_raw, depth_vis_state, ema=0.92)
        depth_color = cv2.resize(depth_color, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        side_by_side_proc = np.hstack([frame, depth_color])

        # ---------------------------
        # YOLO COCO (runs every N frames)
        # ---------------------------
        if frame_idx % COCO_EVERY_N == 0:
            r = coco_model.predict(
                source=frame, conf=COCO_CONF, iou=COCO_IOU,
                imgsz=(PROC_H, PROC_W), verbose=False
            )[0]

            new_boxes, new_labels = [], []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                names = r.names
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    new_boxes.append((x1, y1, x2, y2))
                    new_labels.append(names[cls[i]])

            # update cache
            last_coco_boxes, last_coco_labels = new_boxes, new_labels

            # refresh per-frame variables too
            coco_boxes, coco_labels = last_coco_boxes, last_coco_labels

        # ---------------------------
        # Tint non-road except objects
        # ---------------------------
        base = frame.astype(np.float32) / 255.0
        purple = np.array([0.7, 0.0, 0.7], dtype=np.float32)
        alphaP = 0.45
        non_road = (~road_mask)

        obj_mask = np.zeros((PROC_H, PROC_W), dtype=bool)
        for (x1, y1, x2, y2) in coco_boxes:
            x1i = max(0, min(PROC_W - 1, int(round(x1))))
            y1i = max(0, min(PROC_H - 1, int(round(y1))))
            x2i = max(0, min(PROC_W - 1, int(round(x2))))
            y2i = max(0, min(PROC_H - 1, int(round(y2))))
            obj_mask[y1i:y2i + 1, x1i:x2i + 1] = True

        purple_mask = non_road & (~obj_mask)
        base[purple_mask] = (1 - alphaP) * base[purple_mask] + alphaP * purple
        out = np.clip(base * 255, 0, 255).astype(np.uint8)

        # ---------------------------
        # Traffic lights (SAFE: no label mismatch)
        # ---------------------------
        tl_boxes = [b for b, lab in zip(coco_boxes, coco_labels) if lab == "traffic light"]
        # Optional: dedup traffic lights only (keeps labels safe because we filter first)
        tl_boxes = dedup_detections_xyxy(tl_boxes, iou_thr=0.60)

        for (x1, y1, x2, y2) in tl_boxes:
            state = traffic_light_state_from_bbox(frame, (x1, y1, x2, y2))
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(out, f"TRAFFIC LIGHT: {state}",
                        (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---------------------------
        # BEV
        # ---------------------------
        bev = warp_to_bev(frame, H_img2bev, bev_size)
        if bev.shape[:2] != (bevH, bevW):
            bev = cv2.resize(bev, (bevW, bevH), interpolation=cv2.INTER_LINEAR)

        # Lane detection on BEV
        if LANE_ENABLE and (frame_idx % LANE_EVERY_N == 0):
            lane_overlay_bev, lane_bin, lane_ok = lane_hough_sliding(bev, draw_windows=LANE_DEBUG_DRAW_WINDOWS)
            last_lane_overlay_bev = lane_overlay_bev
            last_lane_ok = lane_ok

        bev_with_lane = overlay_nonzero(bev, last_lane_overlay_bev, alpha=LANE_ALPHA)

        if LANE_ENABLE:
            lane_on_proc = warp_bev_overlay_to_proc(last_lane_overlay_bev, H_bev2img, (PROC_W, PROC_H))
            out = overlay_nonzero(out, lane_on_proc, alpha=LANE_ALPHA)

        writer_bev_avi.write(bev_with_lane)
        if writer_bev_mp4 is not None:
            writer_bev_mp4.write(bev_with_lane)

        # ---------------------------
        # Signs (runs every N frames)
        # ---------------------------
        if frame_idx % SIGN_EVERY_N == 0:
            s = sign_model.predict(
                source=frame, conf=SIGN_CONF, iou=SIGN_IOU,
                imgsz=(PROC_H, PROC_W), verbose=False
            )[0]
            new_sd, new_sl = [], []
            if s.boxes is not None and len(s.boxes) > 0:
                s_xyxy = s.boxes.xyxy.cpu().numpy()
                s_cls = s.boxes.cls.cpu().numpy().astype(int)
                s_names = s.names
                for i in range(len(s_xyxy)):
                    x1, y1, x2, y2 = map(float, s_xyxy[i])
                    new_sd.append((x1, y1, x2, y2))
                    new_sl.append(s_names[s_cls[i]])
            last_sign_dets, last_sign_labels = new_sd, new_sl

        # use last cache every frame
        sign_dets = dedup_detections_xyxy(last_sign_dets, iou_thr=0.70)
        sign_tracks = sign_tracker.update(sign_dets, last_sign_labels)
        sign_tracker.tracks = sign_tracks

        for tr in sign_tracks:
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 150, 255), 2)
            cv2.putText(out, f"Sign {tr.display_id}: {pretty_sign_label(tr.label)}",
                        (int(x1), int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

        # ---------------------------
        # Vehicles + Persons (from COCO cached detections)
        # ---------------------------
        veh_dets, veh_labs = [], []
        for bbox, lab in zip(coco_boxes, coco_labels):
            if lab in vehicle_classes:
                veh_dets.append(bbox)
                veh_labs.append(lab)

        veh_dets = dedup_detections_xyxy(veh_dets, iou_thr=0.70)
        veh_tracks = vehicle_tracker.update(veh_dets, veh_labs)
        vehicle_tracker.tracks = veh_tracks

        person_dets = [bbox for bbox, lab in zip(coco_boxes, coco_labels) if lab == "person"]
        person_dets = dedup_detections_xyxy(person_dets, iou_thr=0.70)
        ped_tracks = person_tracker.update(person_dets)
        ped_tracks = dedup_tracks_by_iou(ped_tracks, iou_thr=0.65)
        person_tracker.tracks = ped_tracks

        dbg_person_dets = len(person_dets)
        dbg_depth_valid = 0

        # Pass 1: collect (yb, depth) for auto invert
        ped_depth_pairs = []
        for tr in ped_tracks:
            x1, y1, x2, y2 = tr.bbox
            yb = y2
            dval = robust_bbox_depth(depth_raw, (x1, y1, x2, y2), PROC_H, PROC_W)
            ped_depth_pairs.append((yb, dval))

        inv_est = auto_invert_depth_for_people(ped_depth_pairs, default_invert=DEPTH_INVERT_DEFAULT)

        if (INVERT_LOCK and not inv_locked):
            inv_smooth = INVERT_EMA * inv_smooth + (1 - INVERT_EMA) * (1.0 if inv_est else 0.0)
            if frame_idx >= INVERT_WARMUP_FRAMES:
                inv_locked = True
        elif not inv_locked:
            inv_smooth = INVERT_EMA * inv_smooth + (1 - INVERT_EMA) * (1.0 if inv_est else 0.0)

        DEPTH_INVERT_FRAME = (inv_smooth >= 0.5) if (inv_locked or frame_idx >= 10) else DEPTH_INVERT_DEFAULT

        # Vehicle intent + draw
        for tr in veh_tracks:
            x1, y1, x2, y2 = tr.bbox
            dval = robust_bbox_depth(depth_raw, (x1, y1, x2, y2), PROC_H, PROC_W)
            tr.depth_hist.append(dval)
            tr.intent = infer_approach_intent(list(tr.depth_hist), invert=DEPTH_INVERT_FRAME, min_n=10, thr=0.03)

            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(out, f"{tr.label} {tr.display_id}: {tr.intent}",
                        (int(x1), int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Pedestrians annotate + mismatch
        for tr in ped_tracks:
            x1, y1, x2, y2 = tr.bbox
            xc = 0.5 * (x1 + x2)
            yb = y2

            side = classify_side_grid(xc, yb, PROC_W, PROC_H, y_split, van_x, base_left_x, base_right_x)
            dist_grid = classify_distance_far_mid_close(yb, PROC_H)

            tr.side_hist.append(side)
            tr.x_hist.append(float(xc))
            tr.y_hist.append(float(yb))

            if dist_grid in ("MID", "CLOSE"):
                tr.last_intent = infer_ped_intent(tr.side_hist, tr.x_hist, tr.y_hist, min_frames=10)
            else:
                tr.last_intent = "UNKNOWN"

            dval = robust_bbox_depth(depth_raw, (x1, y1, x2, y2), PROC_H, PROC_W)
            if dval is not None:
                dbg_depth_valid += 1

            dist_depth = depth_label_from_value(dval, p30, p60, invert=DEPTH_INVERT_FRAME)

            if dist_depth != "UNK":
                total_compared += 1
                if dist_grid != dist_depth:
                    update_grid_depth_mismatch(false_counts, dist_grid, dist_depth)

            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            cv2.putText(out, f"Person {tr.display_id} ({side},{dist_grid},{tr.last_intent})",
                        (int(x1), int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(out, f"DEPTH={dist_depth}",
                        (int(x1), min(PROC_H - 5, int(y2) + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # HUD
        out = draw_hud_full(out, ped_tracks, sign_tracks, veh_tracks,
                            false_counts, total_compared,
                            dbg_depth_valid, dbg_person_dets, DEPTH_INVERT_FRAME, last_lane_ok)

        # upscale to original for writing
        out_up = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_LINEAR)
        side_by_side_up = cv2.resize(side_by_side_proc, (W0 * 2, H0), interpolation=cv2.INTER_LINEAR)

        # write AVI always
        writer_avi.write(out_up)
        writer_rgb_avi.write(side_by_side_up)
        if writer_mp4 is not None:
            writer_mp4.write(out_up)
        if writer_rgb_mp4 is not None:
            writer_rgb_mp4.write(side_by_side_up)

        if SHOW_LIVE:
            cv2.imshow("Annotated", out_up)
            cv2.imshow("RGB|Depth", side_by_side_up)
            cv2.imshow("BEV", bev_with_lane)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # fps cap
        if MAX_FPS_CAP and MAX_FPS_CAP > 0:
            dt = time.time() - t_last
            t_last = time.time()
            target_dt = 1.0 / float(MAX_FPS_CAP)
            if dt < target_dt:
                time.sleep(target_dt - dt)

    cap.release()

    writer_avi.release()
    if writer_mp4 is not None:
        writer_mp4.release()

    writer_rgb_avi.release()
    if writer_rgb_mp4 is not None:
        writer_rgb_mp4.release()

    writer_bev_avi.release()
    if writer_bev_mp4 is not None:
        writer_bev_mp4.release()

    if SHOW_LIVE:
        cv2.destroyAllWindows()

    print("\nSaved (AVI is the correct output on Windows):")
    print(" -", out_avi)
    print(" -", out_rgb_avi)
    print(" -", out_bev_avi)
    if out_mp4: print(" - (optional mp4)", out_mp4)
    if out_rgb_mp4: print(" - (optional mp4)", out_rgb_mp4)
    if out_bev_mp4: print(" - (optional mp4)", out_bev_mp4)
main()