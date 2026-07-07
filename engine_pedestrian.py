"""
engine.py — Billboard / OOH surface recommendation engine.

Adapted from the original street-level image analysis pipeline, tuned
for a pedestrian point-of-view. Given a single street-level / pedestrian-
POV photo, this module:
  1. Runs semantic segmentation (SegFormer, ADE20K classes)
  2. Extracts candidate ad surfaces (existing signboards, blank walls,
     fences/awnings, building facades)
  3. Runs them through a hard-reject gate (size, shape, sky/ground
     position, glass, occlusion, occupied-facade checks, flyover checks...)
  4. Scores every surface that survives the gate (occlusion, pedestrian
     sightline position, surface blankness/flatness, size, shape)
  5. Returns a ranked list of PREMIUM / VIABLE / AVOID sites plus an
     annotated image and human-readable "why recommended" bullets.

This is the same scoring logic as the notebook — just refactored into
importable functions instead of top-level notebook cells, and with a
video/frame-sampling-free "why recommended" bullet generator for the
single-image use case.
"""

import math
import warnings
import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

# ── ADE20K class IDs ─────────────────────────────────────────────────────────
AD_IDS         = {43, 100, 123, 130, 144}   # signboard, poster, trade_name, screen, bulletin_board
WALL_IDS       = {0, 42}                     # wall, column/pillar (e.g. flyover/underpass pillars — flat, structural, often already painted/branded)
FENCE_IDS      = {32, 38, 86, 88, 106}       # fence, railing, awning, booth, canopy
BUILDING_IDS   = {1, 25, 48}                 # building, house, skyscraper
ALL_SURFACE_IDS = AD_IDS | WALL_IDS | FENCE_IDS | BUILDING_IDS

BRIDGE_FLYOVER_IDS = {149, 141, 53, 61}

TREE_IDS       = {4, 5, 17, 66}
WIRE_POLE_IDS  = {67, 126}
SIGN_IDS       = {22}
LIGHT_IDS      = {21}
PERSON_IDS     = {12, 78}
VEHICLE_IDS    = {20, 80, 83, 102, 103, 116, 127}   # car, bus, truck, van, ship, minibike, bicycle
OCCLUDER_IDS   = TREE_IDS | WIRE_POLE_IDS | SIGN_IDS | LIGHT_IDS | VEHICLE_IDS | PERSON_IDS

WINDOW_IDS     = {9}
DOOR_IDS       = {58}
GLASS_IDS      = WINDOW_IDS | DOOR_IDS
SUBTRACT_IDS   = PERSON_IDS | VEHICLE_IDS

SKY_IDS        = {2}
ROAD_IDS       = {3, 6, 11, 52}

CLASS_PALETTE = {
    0:  ("wall",           (255, 100, 100)),
    42: ("pillar",          (255, 100, 100)),
    1:  ("building",       (255,  60, 130)),
    25: ("house",          (255, 160,  60)),
    48: ("skyscraper",     (200,   0, 255)),
    43: ("signboard",      (  0, 220,  40)),
    100:("poster",         (  0, 200, 255)),
    123:("trade_name",     (120, 255, 120)),
    130:("screen",         (255, 255,   0)),
    144:("bulletin_board", (  0, 255, 220)),
    32: ("fence",          ( 80, 220, 255)),
    38: ("railing",        (160, 160, 255)),
    86: ("awning",         (255, 170,   0)),
    88: ("booth",          (170, 255,  80)),
    106:("canopy",         (255,  80, 220)),
}

# ── Resolution normalization ─────────────────────────────────────────────
# Every absolute-pixel constant below (MIN_W_PX, MIN_H_PX, the morphology
# kernel sizes, the 60x40px sub-region floors, etc.) was tuned against
# photos in the ~1000-1300px-wide range. A phone photo saved/compressed
# down to a much smaller width (e.g. 450px) shrinks a real signboard's
# footprint in absolute pixels well below these floors, even though its
# *share of the frame* is identical — so it silently fails candidate
# extraction and never gets isolated from the building/wall behind it.
# Rather than rewrite every constant to be resolution-relative (and risk
# subtly changing tuned behaviour), low-resolution uploads are upscaled
# once, before segmentation, to this canonical width; every downstream
# heuristic then sees pixel counts consistent with what it was tuned on.
ANALYSIS_MIN_WIDTH = 1024

# ── Thresholds ────────────────────────────────────────────────────────────
MIN_AREA_PCT       = 0.018
MIN_W_PX           = 70
MIN_H_PX           = 50
MIN_WH_RATIO       = 0.40
MAX_WH_RATIO       = 5.5
MAX_WH_RATIO_AD    = 8.0
MIN_SOLIDITY       = 0.55

SKY_TOP_NORM         = 0.10
GND_BOT_NORM         = 0.82
PEDESTRIAN_EYE_NORM  = 0.42   # normalized frame-height band where a person's
                              # eye-level naturally falls in a hand-held
                              # street photo — surfaces centered here read
                              # best to someone walking past on foot

FLYOVER_WH_MIN     = 3.8
FLYOVER_Y_MAX_NORM = 0.52
SKY_BELOW_RATIO    = 0.10

GLASS_PCT_REJECT   = 0.10
EDGE_DENSITY_REJECT= 0.22
TEXTURE_VAR_REJECT = 55.0

OCC_PCT_REJECT     = 0.20

COLOR_STD_REJECT   = 48.0
GRADIENT_MAG_REJECT= 30.0

MAX_SKY_OVERLAP    = 0.25

BUILDING_MAX_GLASS       = 0.04
BUILDING_MAX_DOOR        = 0.03
BUILDING_MAX_EDGE        = 0.14
BUILDING_MIN_BLANK       = 0.62
BUILDING_MAX_HEIGHT_NORM = 0.75

SPLIT_AT_PCT              = 0.12
SPLIT_EDGE_DENSITY_THRESH = 0.10
SPLIT_DOOR_THRESH         = 0.01
SPLIT_GLASS_THRESH        = 0.02

W_OCC      = 0.35
W_POS      = 0.25
W_SURFACE  = 0.25
W_SIZE     = 0.10
W_SHAPE    = 0.05

THR_PREMIUM = 0.72
THR_VIABLE  = 0.52
MAX_REALISTIC_SCORE = 0.96

BUILDING_SCORE_CAP      = THR_PREMIUM - 0.01

REC_COL = {
    "PREMIUM": (0, 210, 80),
    "VIABLE":  (255, 165, 0),
}


# ═════════════════════════════════════════════════════════════════════════
# Model loading (lazy — only happens once, on first analyze() call)
# ═════════════════════════════════════════════════════════════════════════
_model = None
_processor = None
_device = None


def _load_model():
    global _model, _processor, _device
    if _model is not None:
        return
    import torch

    # Use the generic Auto* classes rather than Segformer-specific ones.
    # Different `transformers` releases have, at various points, moved or
    # renamed the model-specific classes (e.g. SegformerImageProcessor),
    # which breaks a hardcoded import on some installed versions. The
    # Auto* loaders resolve to the right underlying class from the model's
    # own config, so they work across transformers versions.
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()


def segment(img_pil):
    """Runs semantic segmentation on a PIL image already held in memory.

    Takes a PIL.Image directly (rather than re-reading from disk) so the
    seg_map is guaranteed to be pixel-aligned with whatever array
    analyze_image() is actually using downstream — including the resized
    copy analyze_image() builds for small/low-resolution uploads (see
    ANALYSIS_MIN_WIDTH below)."""
    import torch
    import torch.nn.functional as F
    _load_model()
    img = img_pil.convert("RGB")
    W, H = img.size
    inp = {k: v.to(_device) for k, v in _processor(images=img, return_tensors="pt").items()}
    with torch.no_grad():
        logits = _model(**inp).logits
    seg = (F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
             .argmax(1).squeeze().cpu().numpy().astype(np.int32))
    return seg


# ═════════════════════════════════════════════════════════════════════════
# Low-level analysis helpers
# ═════════════════════════════════════════════════════════════════════════

def morph_clean(mask, close=9, opn=3):
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opn, opn))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kc)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)


def remove_foreground(mask, seg_map):
    fg = np.isin(seg_map, list(SUBTRACT_IDS)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.dilate(fg, k, iterations=2)
    return np.where(fg == 1, 0, mask).astype(np.uint8)


def polygon_from_mask(mask_crop, offset_x, offset_y, epsilon_ratio=0.012, max_points=16):
    """
    Turns a binary mask crop into a clean, simplified polygon that follows
    the actual surface silhouette (e.g. a wall's receding top edge in
    perspective, or the notch where it meets a gate/building) rather than
    the wide rectangle that used to be drawn around it.

    - A light morphological close + blur/threshold pass first smooths the
      stair-stepped pixel edges that segmentation masks produce, so the
      polygon doesn't look jagged or noisy.
    - cv2.approxPolyDP then reduces the contour to a small, clean set of
      vertices (capped at max_points), backing off in coarseness until
      it's under the cap, so outlines stay legible instead of tracing
      every tiny wiggle in the mask boundary.

    Returns a list of [x, y] points in full-image coordinates, or None if
    the crop has no usable contour (caller should fall back to a rectangle).
    """
    m = (mask_crop > 0).astype(np.uint8)
    if m.sum() < 30:
        return None

    try:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m * 255, cv2.MORPH_CLOSE, k)
        m = cv2.GaussianBlur(m, (7, 7), 0)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        lg = max(contours, key=cv2.contourArea)
        if cv2.contourArea(lg) < 30:
            return None

        perimeter = cv2.arcLength(lg, True)
        if perimeter <= 0:
            return None

        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(lg, epsilon, True)
        tries = 0
        while len(approx) > max_points and tries < 8:
            epsilon *= 1.35
            approx = cv2.approxPolyDP(lg, epsilon, True)
            tries += 1
    except Exception:
        # Fall back to a rectangle rather than let one bad crop crash
        # the whole analysis — tight_rect already handles poly=None.
        return None

    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None

    pts = pts + np.array([offset_x, offset_y])
    return pts.tolist()


def tight_rect(mask):
    if mask.sum() == 0:
        return None, None, 0, 0.0
    col_occ = mask.sum(axis=0).astype(float) / mask.shape[0]
    row_occ = mask.sum(axis=1).astype(float) / mask.shape[1]
    col_valid = np.where(col_occ > 0.05)[0]
    row_valid = np.where(row_occ > 0.05)[0]
    if len(col_valid) == 0 or len(row_valid) == 0:
        return None, None, 0, 0.0
    x1, x2 = int(col_valid.min()), int(col_valid.max())
    y1, y2 = int(row_valid.min()), int(row_valid.max())
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None, None, 0, 0.0
    crop = mask[y1:y2, x1:x2]
    area = int(crop.sum())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        lg = max(contours, key=cv2.contourArea)
        ha = cv2.contourArea(cv2.convexHull(lg))
        solidity = cv2.contourArea(lg) / ha if ha > 0 else 0.0

    # Prefer a clean polygon that traces the surface's actual silhouette;
    # fall back to the old rectangle only if the contour isn't usable
    # (e.g. a near-empty or degenerate crop).
    poly = polygon_from_mask(crop, x1, y1)
    if poly is None:
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    return poly, [x1, y1, w, h], area, round(solidity, 3)


def edge_density_in_box(img_bgr, bbox, seg_map=None, class_ids=None):
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0:
        return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    if seg_map is not None and class_ids is not None:
        roi_seg = seg_map[y:y+h, x:x+w]
        mask = np.isin(roi_seg, list(class_ids))
        if mask.sum() < 50:
            return 1.0
        return float((edges > 0)[mask].sum()) / mask.sum()
    return float(edges.sum() / 255) / edges.size


def colour_uniformity(img_bgr, bbox, seg_map=None, class_ids=None):
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w].astype(float)
    if crop.size == 0:
        return 255.0
    if seg_map is not None and class_ids is not None:
        roi_seg = seg_map[y:y+h, x:x+w]
        mask = np.isin(roi_seg, list(class_ids))
        if mask.sum() < 50:
            return 255.0
        return float(np.mean([crop[:, :, c][mask].std() for c in range(3)]))
    return float(np.mean([crop[:, :, c].std() for c in range(3)]))


def laplacian_variance(img_bgr, bbox):
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0:
        return 9999.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_gradient_magnitude(img_bgr, bbox):
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0:
        return 999.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean())


def glass_ratio_in_box(bbox, seg_map, class_ids=None):
    x, y, w, h = bbox
    roi = seg_map[y:y+h, x:x+w]
    if roi.size == 0:
        return 1.0
    if class_ids is not None:
        mask = np.isin(roi, list(class_ids))
        if mask.sum() < 50:
            return 0.0
        return float(np.isin(roi, list(GLASS_IDS))[mask].sum()) / mask.sum()
    return float(np.isin(roi, list(GLASS_IDS)).sum()) / roi.size


def door_ratio_in_box(bbox, seg_map, class_ids=None):
    x, y, w, h = bbox
    roi = seg_map[y:y+h, x:x+w]
    if roi.size == 0:
        return 1.0
    if class_ids is not None:
        mask = np.isin(roi, list(class_ids))
        if mask.sum() < 50:
            return 0.0
        return float((roi == 58)[mask].sum()) / mask.sum()
    return float((roi == 58).sum()) / roi.size


def occluder_ratio_in_box(bbox, seg_map):
    x, y, w, h = bbox
    roi = seg_map[y:y+h, x:x+w]
    if roi.size == 0:
        return 1.0
    return float(np.isin(roi, list(OCCLUDER_IDS)).sum()) / roi.size


def sky_overlap_ratio(bbox, frame_h, seg_map=None):
    x, y, w, h = bbox
    if seg_map is not None:
        roi = seg_map[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0
        return float(np.isin(roi, list(SKY_IDS)).sum()) / roi.size
    sky_limit = int(SKY_TOP_NORM * frame_h)
    return max(0, min(y + h, sky_limit) - y) / max(h, 1)


def is_flyover_underside(bbox, seg_map, img_bgr, frame_h, frame_w, sky_mask, road_mask, class_id=None):
    if class_id is not None and class_id in AD_IDS:
        return False, ""
    x, y, w, h = bbox
    wh_ratio = w / max(h, 1)
    top_norm = y / frame_h
    if wh_ratio < FLYOVER_WH_MIN:
        return False, ""
    if top_norm > FLYOVER_Y_MAX_NORM:
        return False, ""
    below_y = min(y + h + int(0.05 * frame_h), frame_h)
    below_roi_sky = sky_mask[y+h:below_y, x:x+w]
    below_roi_road = road_mask[y+h:below_y, x:x+w]
    if below_roi_sky.size > 0:
        sky_below_frac = below_roi_sky.mean()
        road_below_frac = below_roi_road.mean()
    else:
        sky_below_frac, road_below_frac = 0.0, 0.0
    if sky_below_frac > SKY_BELOW_RATIO:
        return True, f"flyover: wide (w/h={wh_ratio:.1f}), sky below ({sky_below_frac*100:.0f}%)"
    if road_below_frac > 0.25 and wh_ratio > 4.5:
        return True, f"flyover: wide (w/h={wh_ratio:.1f}), road directly below ({road_below_frac*100:.0f}%)"
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean()
        gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean()
        h_dominance = gy / (gx + 1e-5)
        if h_dominance > 2.0 and wh_ratio > 4.0:
            return True, f"flyover: horizontal structure (h_dom={h_dominance:.1f}), wide (w/h={wh_ratio:.1f})"
    return False, ""


def detect_window_grid(img_bgr, bbox, seg_map=None, class_ids=None, min_blobs=3):
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0 or w < 40 or h < 40:
        return False, ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if seg_map is not None and class_ids is not None:
        roi_seg = seg_map[y:y+h, x:x+w]
        surf_mask = np.isin(roi_seg, list(class_ids)).astype(np.uint8)
        if surf_mask.sum() < 200:
            return False, ""
    else:
        surf_mask = np.ones_like(gray, dtype=np.uint8)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mean_local = cv2.boxFilter(blur.astype(np.float32), -1, (17, 17))
    diff = np.abs(blur.astype(np.float32) - mean_local)
    _, contrast_blobs = cv2.threshold(diff.astype(np.uint8), 14, 255, cv2.THRESH_BINARY)
    contrast_blobs = contrast_blobs * surf_mask * 255 if surf_mask.max() <= 1 else contrast_blobs
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    contrast_blobs = cv2.morphologyEx(contrast_blobs, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(contrast_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_area = w * h
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        area_frac = area / box_area
        if area_frac < 0.0004 or area_frac > 0.05:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        ratio = rw / max(rh, 1)
        if ratio < 0.35 or ratio > 2.8:
            continue
        rect_area = rw * rh
        if rect_area == 0:
            continue
        fill_ratio = area / rect_area
        if fill_ratio < 0.45:
            continue
        candidates.append((x + rx, y + ry, rw, rh))
    if len(candidates) < min_blobs:
        return False, ""
    centers_y = sorted(cy + ch/2 for (_, cy, _, ch) in candidates)
    rows, current_row = [], [centers_y[0]]
    row_tol = max(h * 0.06, 12)
    for cy in centers_y[1:]:
        if cy - current_row[-1] <= row_tol:
            current_row.append(cy)
        else:
            rows.append(current_row)
            current_row = [cy]
    rows.append(current_row)
    multi_blob_rows = sum(1 for r in rows if len(r) >= 2)
    if multi_blob_rows >= 1 and len(candidates) >= min_blobs:
        return True, f"window grid detected ({len(candidates)} window-like openings, {multi_blob_rows} aligned rows)"
    if len(candidates) >= min_blobs + 2:
        return True, f"multiple window-like openings detected ({len(candidates)})"
    return False, ""


def detect_balcony_railings(img_bgr, bbox, seg_map=None, class_ids=None, min_bands=4):
    """Rejects facades whose apparent 'blankness' is actually balcony
    railings, grillwork, hanging laundry lines, or plant shelves — all of
    which read as long, fairly straight, evenly-spaced horizontal edges
    rather than the window-grid rectangles detect_window_grid looks for.
    Common on residential buildings and easy to miss for that reason,
    especially now that pedestrian photos are taken close enough to
    resolve this kind of detail clearly."""
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0 or w < 40 or h < 40:
        return False, ""
    if seg_map is not None and class_ids is not None:
        roi_seg = seg_map[y:y+h, x:x+w]
        surf_mask = np.isin(roi_seg, list(class_ids))
        if surf_mask.sum() < 200:
            return False, ""
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(int(w * 0.22), 15),
                                 minLineLength=max(int(w * 0.22), 15), maxLineGap=6)
        if lines is None:
            return False, ""
        row_centers = []
        # HoughLinesP normally returns shape (N,1,4), but flatten defensively
        # in case a given OpenCV build/config returns (N,4) instead — either
        # way np.ravel(l) always yields the 4 coordinates as a flat array.
        for l in lines:
            lx1, ly1, lx2, ly2 = np.ravel(l)[:4]
            length = math.hypot(float(lx2) - float(lx1), float(ly2) - float(ly1))
            if length < w * 0.20:
                continue
            angle = abs(math.degrees(math.atan2(float(ly2) - float(ly1), float(lx2) - float(lx1))))
            if angle < 12 or angle > 168:
                row_centers.append((float(ly1) + float(ly2)) / 2.0)
    except Exception:
        # This is a soft heuristic on top of the main gate — if anything
        # about a particular crop trips up line detection, skip the check
        # rather than failing the whole analysis over one candidate.
        return False, ""
    if len(row_centers) < min_bands:
        return False, ""
    row_centers.sort()
    distinct_bands = 1
    for i in range(1, len(row_centers)):
        if row_centers[i] - row_centers[i-1] > max(h * 0.035, 6):
            distinct_bands += 1
    if distinct_bands >= min_bands:
        return True, (f"balcony/railing pattern detected ({distinct_bands} evenly-spaced "
                       f"horizontal bands, {len(row_centers)} long horizontal edges)")
    return False, ""


def facade_periodicity_score(img_bgr, bbox):
    """Detects the repeating window/balcony grid of an occupied facade via
    autocorrelation of the column and row brightness/gradient profiles,
    rather than local edge density or per-pixel segmentation class.

    This matters specifically for buildings seen from far away (a normal
    pedestrian photo of a multi-story block across the street): at that
    distance the individual window openings shrink to a few px, get
    softened by JPEG compression and haze, and often blur below the
    SegFormer model's resolution entirely — so per-window edge/contrast/
    glass-class signals (which is what every other check here relies on)
    come out artificially weak or don't even survive as their own
    segmentation class. But the *regular spacing* of windows/balconies
    survives distance and blur just fine, because it's a low-frequency,
    whole-facade pattern rather than a fine-detail one — which is also
    exactly how a person recognizes "that's a lived-in building" from
    across the street without being able to make out any single window.
    Returns a 0-1 confidence that this bbox shows a periodic facade.
    """
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0 or w < 40 or h < 40:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    def periodicity(profile):
        profile = profile - profile.mean()
        if profile.std() < 1e-3:
            return 0.0
        n = len(profile)
        f = np.fft.rfft(profile * np.hanning(n))
        power = np.abs(f) ** 2
        lo_cut = max(2, n // 40)
        if lo_cut >= len(power) - 2:
            return 0.0
        band = power[lo_cut:]
        if band.sum() < 1e-6:
            return 0.0
        peak = band.max()
        return float(np.clip(peak / (band.sum() / len(band)) / 25.0, 0, 1))

    col_profile = gray.mean(axis=0)  # vertical grid lines (window columns)
    row_profile = gray.mean(axis=1)  # horizontal bands (window/balcony rows)
    return float(max(periodicity(col_profile), periodicity(row_profile)))


def is_occupied_building_facade(bbox, seg_map, img_bgr, active, is_genuine_building=False):
    """Rejects surfaces that are actually occupied storefronts, homes, or
    other 'lived-in' facades: window grids, doors, balconies, AC units,
    shop displays.

    IMPORTANT: every ratio/density check below is measured over the FULL
    bbox crop (class_ids=None), not filtered down to only the pixels the
    segmentation model itself labeled 'building'. Windows, doors, and
    railings are their own separate ADE20K classes — filtering to
    "building-only" pixels silently excludes precisely the pixels that
    would prove a facade is occupied, which was the bug that let a whole
    apartment block through as a 'blank wall'. We only look at `seg_map`
    directly (unfiltered class lookups) to see whatever the model
    actually resolved, wherever it landed."""
    if not active:
        return False, ""
    x, y, w, h = bbox

    d_ratio = door_ratio_in_box(bbox, seg_map, None)
    if d_ratio > BUILDING_MAX_DOOR:
        return True, f"occupied surface: door visible ({d_ratio*100:.1f}%) — entrance, not a hoarding"
    g_ratio = glass_ratio_in_box(bbox, seg_map, None)
    g_reject = BUILDING_MAX_GLASS if not is_genuine_building else min(BUILDING_MAX_GLASS, 0.02)
    if g_ratio > g_reject:
        return True, f"occupied surface: {g_ratio*100:.0f}% glass/windows — storefront, not a hoarding"
    e_dens = edge_density_in_box(img_bgr, bbox, seg_map, None)
    if e_dens > BUILDING_MAX_EDGE:
        return True, f"cluttered surface: edge density {e_dens:.2f} (window/display/AC grid)"
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size > 0 and h > 60:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        n_strips = 5
        strip_h = h // n_strips
        strip_densities = []
        for si in range(n_strips):
            strip = edges[si*strip_h:(si+1)*strip_h, :]
            d = float(strip.sum() / 255) / (strip.size + 1e-5)
            strip_densities.append(d)
        strip_arr = np.array(strip_densities)
        high_strips = (strip_arr > 0.08).sum()
        if high_strips >= 4:
            return True, f"cluttered surface: {high_strips}/5 strips busy (window/display grid pattern)"
    c_std = colour_uniformity(img_bgr, bbox, seg_map, None)
    if c_std > COLOR_STD_REJECT:
        return True, f"occupied/cluttered surface: high colour variance ({c_std:.1f}) — signage/products/people"
    has_windows, win_reason = detect_window_grid(img_bgr, bbox, seg_map, None)
    if has_windows:
        return True, f"occupied surface: {win_reason}"
    has_rail, rail_reason = detect_balcony_railings(img_bgr, bbox, seg_map, None)
    if has_rail:
        return True, f"occupied surface: {rail_reason}"

    if is_genuine_building:
        # NOTE: an earlier version of this check also ran
        # facade_periodicity_score() as a hard-reject trigger (looking for
        # the repeating window/balcony grid of a residential facade via
        # FFT peak analysis on brightness profiles). Testing found it
        # produced false positives on brick coursing and random textured/
        # stained walls — plain periodic or quasi-periodic texture isn't
        # by itself proof of windows, and this pipeline has no way to
        # tell window-spacing-scale periodicity apart from brick/tile-
        # scale periodicity without a distance estimate. Left out of the
        # gate for now rather than risk rejecting legitimate blank/brick
        # compound walls; the direct glass/door/edge/colour/window-grid/
        # railing checks above (now correctly unmasked) plus the blank-
        # score bar below are what's actually carrying this gate.

        # Genuine building-class candidates get held to a higher blank-
        # surface bar than the checks above catch on their own — a
        # facade can dodge every specific glass/door/window/railing
        # trigger individually and still just be a busy, lived-in wall
        # (mixed tile patterns, stains, plant shelves, wiring) rather
        # than a clean paintable surface.
        bscore = blank_wall_score(img_bgr, bbox, seg_map, None)
        if bscore < BUILDING_MIN_BLANK:
            return True, (f"textured/cluttered building facade (blank-surface score "
                           f"{bscore*100:.0f}%) — below the minimum for a paintable hoarding")
    return False, ""


def blank_wall_score(img_bgr, bbox, seg_map=None, class_ids=None):
    c_std = colour_uniformity(img_bgr, bbox, seg_map, class_ids)
    e_dens = edge_density_in_box(img_bgr, bbox, seg_map, class_ids)
    lap = laplacian_variance(img_bgr, bbox)
    grad = mean_gradient_magnitude(img_bgr, bbox)
    s_colour = np.clip(1.0 - (c_std - 15) / 60, 0, 1)
    s_edge = np.clip(1.0 - (e_dens - 0.02) / 0.18, 0, 1)
    s_lap = np.clip(1.0 - (lap - 50) / 500, 0, 1)
    s_grad = np.clip(1.0 - (grad - 10) / 35, 0, 1)
    return round(float(0.30*s_colour + 0.30*s_edge + 0.25*s_lap + 0.15*s_grad), 3)


def vertical_position_score(bbox, frame_h):
    x, y, w, h = bbox
    cy_norm = (y + h / 2) / frame_h
    if cy_norm < SKY_TOP_NORM or cy_norm > GND_BOT_NORM:
        return 0.0
    return round(float(math.exp(-0.5 * ((cy_norm - PEDESTRIAN_EYE_NORM) / 0.22) ** 2)), 3)


def size_score(area_pct_0_1, is_ad):
    if is_ad:
        lo, hi = 0.004, 0.07
    else:
        lo, hi = 0.018, 0.18
    if area_pct_0_1 < lo:
        return area_pct_0_1 / lo * 0.3
    if area_pct_0_1 > hi:
        return max(0.25, 1.0 - (area_pct_0_1 - hi) / hi)
    return 1.0


def shape_score(bbox):
    _, _, w, h = bbox
    r = w / max(h, 1)
    if 2.0 <= r <= 3.5:
        return 1.00
    elif 1.5 <= r < 2.0:
        return 0.80
    elif 3.5 < r <= 4.5:
        return 0.55
    elif 1.0 <= r < 1.5:
        return 0.60
    elif 0.6 <= r < 1.0:
        return 0.40
    else:
        return 0.15


def natural_col_splits(mask, x, w, min_gap=6):
    col = mask[:, x:x+w].sum(0).astype(float)
    if col.max() == 0:
        return []
    col /= col.max()
    splits, in_g, gs = [], False, 0
    for i, empty in enumerate(col < 0.10):
        if empty and not in_g:
            in_g, gs = True, i
        elif not empty and in_g:
            in_g = False
            if i - gs >= min_gap:
                splits.append(x + gs + (i - gs) // 2)
    return splits


def natural_row_splits(mask, y, h, min_gap=6):
    row = mask[y:y+h, :].sum(1).astype(float)
    if row.max() == 0:
        return []
    row /= row.max()
    splits, in_g, gs = [], False, 0
    for i, empty in enumerate(row < 0.10):
        if empty and not in_g:
            in_g, gs = True, i
        elif not empty and in_g:
            in_g = False
            if i - gs >= min_gap:
                splits.append(y + gs + (i - gs) // 2)
    return splits


def detect_screen_subregions(comp_mask, bbox, img_bgr, img_hsv, min_area_frac=0.15):
    x, y, w, h = bbox
    if w < MIN_W_PX or h < MIN_H_PX:
        return []
    sub_comp = comp_mask[y:y+h, x:x+w]
    hsv_crop = img_hsv[y:y+h, x:x+w]
    gray_crop = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sat = hsv_crop[:, :, 1].astype(float)
    val = hsv_crop[:, :, 2].astype(float)
    edges = cv2.Canny(cv2.GaussianBlur(gray_crop, (5, 5), 0), 40, 120)
    edge_density_map = cv2.boxFilter((edges > 0).astype(np.float32), -1, (21, 21))
    colourful = sat > 100
    bright_busy = (val > 210) & (edge_density_map > 0.04)
    screen_like = (colourful | bright_busy).astype(np.uint8)
    screen_like = screen_like * sub_comp
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    screen_like = cv2.morphologyEx(screen_like, cv2.MORPH_CLOSE, k)
    screen_like = cv2.morphologyEx(screen_like, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cnts, _ = cv2.findContours(screen_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_frac * w * h:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        rect_area = rw * rh
        if rect_area == 0:
            continue
        fill_ratio = area / rect_area
        if rw * rh > 0.6 * w * h:
            continue
        if fill_ratio > 0.65 and rw >= MIN_W_PX and rh >= MIN_H_PX:
            boxes.append((x + rx, y + ry, rw, rh))
    return boxes


def detect_dark_signage_subregions(comp_mask, bbox, img_bgr, img_hsv, min_area_frac=0.05):
    x, y, w, h = bbox
    if w < MIN_W_PX or h < MIN_H_PX:
        return []
    sub_comp = comp_mask[y:y+h, x:x+w]
    hsv_crop = img_hsv[y:y+h, x:x+w]
    gray = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sat = hsv_crop[:, :, 1].astype(float)
    mean_local = cv2.boxFilter(gray.astype(np.float32), -1, (15, 15))
    sq_local = cv2.boxFilter((gray.astype(np.float32))**2, -1, (15, 15))
    local_std = np.sqrt(np.clip(sq_local - mean_local**2, 0, None))
    saturated = sat > 70
    high_contrast = local_std > 18
    signage_like = (saturated & high_contrast).astype(np.uint8)
    signage_like = signage_like * sub_comp
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    signage_like = cv2.morphologyEx(signage_like, cv2.MORPH_CLOSE, k_close)
    signage_like = cv2.morphologyEx(signage_like, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cnts, _ = cv2.findContours(signage_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        area_frac = area / (w * h)
        if area < min_area_frac * w * h:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rw * rh > 0.4 * w * h:
            continue
        if rw < 60 or rh < 40:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.55:
            continue
        boxes.append((x + rx, y + ry, rw, rh))
    return boxes


def color_edge_best_col_split(img_bgr, bbox, seg_map=None, class_ids=None, min_w=None):
    x, y, w, h = bbox
    min_w = min_w or MIN_W_PX
    if w < 2 * min_w:
        return None
    crop = img_bgr[y:y+h, x:x+w].astype(np.float32)
    if seg_map is not None and class_ids is not None:
        roi_seg = seg_map[y:y+h, x:x+w]
        mask = np.isin(roi_seg, list(class_ids))
    else:
        mask = np.ones((h, w), dtype=bool)
    col_means = np.full((w, 3), np.nan, dtype=np.float32)
    min_pix = max(5, int(h * 0.05))
    for cidx in range(w):
        colmask = mask[:, cidx]
        if colmask.sum() < min_pix:
            continue
        col_means[cidx] = crop[:, cidx][colmask].mean(axis=0)
    valid = ~np.isnan(col_means[:, 0])
    if valid.sum() < 10:
        return None
    idxs = np.arange(w)
    for c in range(3):
        col_means[:, c] = np.interp(idxs, idxs[valid], col_means[valid, c])
    diffs = np.abs(np.diff(col_means, axis=0)).sum(axis=1)
    diffs = cv2.GaussianBlur(diffs.reshape(1, -1), (1, 9), 0).flatten()
    margin = max(int(w * 0.08), min_w // 2)
    diffs[:margin] = 0
    diffs[-margin:] = 0
    if diffs.max() < 8:
        return None
    return x + int(np.argmax(diffs))


def recursive_colour_split(comp_mask, bbox, seg_map, img_bgr, class_ids=None, depth=0, max_depth=4, out=None):
    if out is None:
        out = []
    if class_ids is None:
        class_ids = BUILDING_IDS
    x, y, w, h = bbox
    c_std = colour_uniformity(img_bgr, bbox, seg_map, class_ids)
    if c_std <= COLOR_STD_REJECT or depth >= max_depth or w < 2 * MIN_W_PX:
        out.append((comp_mask, bbox))
        return out
    split_x = color_edge_best_col_split(img_bgr, bbox, seg_map, class_ids)
    if split_x is None or split_x <= x + MIN_W_PX or split_x >= x + w - MIN_W_PX:
        out.append((comp_mask, bbox))
        return out
    left_mask = comp_mask.copy(); left_mask[:, split_x:] = 0
    right_mask = comp_mask.copy(); right_mask[:, :split_x] = 0
    _, lbbox, _, _ = tight_rect(left_mask)
    _, rbbox, _, _ = tight_rect(right_mask)
    if lbbox is not None:
        recursive_colour_split(left_mask, lbbox, seg_map, img_bgr, class_ids, depth + 1, max_depth, out)
    if rbbox is not None:
        recursive_colour_split(right_mask, rbbox, seg_map, img_bgr, class_ids, depth + 1, max_depth, out)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Hard-reject gate
# ═════════════════════════════════════════════════════════════════════════

def dominant_field_color_ratio(bbox, img_hsv, seg_map=None, class_ids=None):
    """Generalizes the old green/blue-only check to ANY single background
    colour. Route/place-name/tourist signs are built from exactly two
    ingredients everywhere in the world — one flat field colour (green,
    blue, brown, yellow, white...) plus light/dark text — no matter which
    colour the road authority happens to use. Commercial ads are far more
    likely to carry photographic content, logos, or product shots that
    introduce additional, unrelated colours. Returns
    (field_ratio, achromatic_ratio, other_ratio, field_mean_sat).

    field_mean_sat is the mean HSV saturation of the pixels that make up
    the dominant colour field. Retroreflective road-sign paint (NH green,
    city blue, heritage brown) is strongly saturated; a pale/washed-out
    photographic sky or a soft product-photo gradient can independently
    end up "flat" (low hue variance) without being an actual painted sign
    colour, so field_ratio/ach_ratio alone under-distinguish a real-estate
    ad's sky-and-greenery creative from genuine sign paint."""
    x, y, w, h = bbox
    hsv_crop = img_hsv[y:y+h, x:x+w]
    if hsv_crop.size == 0:
        return 0.0, 0.0, 1.0, 0.0
    if seg_map is not None and class_ids is not None:
        roi_seg = seg_map[y:y+h, x:x+w]
        mask = np.isin(roi_seg, list(class_ids))
        if mask.sum() < 50:
            mask = np.ones(hsv_crop.shape[:2], dtype=bool)
    else:
        mask = np.ones(hsv_crop.shape[:2], dtype=bool)
    hue = hsv_crop[:, :, 0][mask].astype(float)
    sat = hsv_crop[:, :, 1][mask].astype(float)
    val = hsv_crop[:, :, 2][mask].astype(float)
    total = max(len(hue), 1)
    # "Achromatic" is meant to capture crisp white/black TEXT and borders,
    # not desaturated photographic content. A product photo (an AC unit,
    # a solar panel, a grey appliance) is also low-saturation, but it sits
    # at MID brightness with soft gradients — real sign text/borders sit
    # at the brightness extremes (near-white or near-black). Without this
    # distinction, a commercial ad with a grey/monochrome product photo
    # gets misread as "mostly flat text" just like a government sign.
    achromatic = (sat < 40) & ((val > 200) | (val < 60))
    ach_ratio = float(achromatic.sum()) / total
    saturated = (sat >= 40) & (val > 30)
    if saturated.sum() < 0.05 * total:
        return 0.0, ach_ratio, max(0.0, 1.0 - ach_ratio), 0.0
    edges = np.linspace(0, 180, 19)  # 18 bins of 10 degrees, hue is 0-179 in OpenCV
    hist, _ = np.histogram(hue[saturated], bins=edges)
    dom_bin = int(np.argmax(hist))
    lo, hi = edges[dom_bin] - 10, edges[dom_bin] + 20  # dominant bin +/- 1 neighbour
    hs = hue[saturated]
    in_field = ((hs >= lo) & (hs <= hi)) | ((hs + 180 >= lo) & (hs + 180 <= hi)) | ((hs - 180 >= lo) & (hs - 180 <= hi))
    field_ratio = float(in_field.sum()) / total
    other_ratio = max(0.0, 1.0 - field_ratio - ach_ratio)
    field_mean_sat = float(sat[saturated][in_field].mean()) if in_field.sum() > 0 else 0.0
    return field_ratio, ach_ratio, other_ratio, field_mean_sat


def road_below_bbox_ratio(bbox, road_mask, frame_h, frame_w, pad_frac=0.08):
    """How much of the strip of frame directly under this box is road —
    the telltale sign that a surface is mounted OVER traffic (a gantry
    sign) rather than beside/on a building."""
    x, y, w, h = bbox
    below_y2 = min(y + h + int(pad_frac * frame_h), frame_h)
    below = road_mask[y+h:below_y2, x:x+w]
    if below.size == 0:
        return 0.0
    return float(below.mean())


def pole_support_below(bbox, seg_map, frame_h, pad_frac=0.10):
    """Checks for a pole/wire directly under the surface — the support
    structure typical of a roadside signpost, as opposed to a hoarding
    mounted flat against a wall or building."""
    x, y, w, h = bbox
    below_y2 = min(y + h + int(pad_frac * frame_h), frame_h)
    below = seg_map[y+h:below_y2, x:x+w]
    if below.size == 0:
        return False
    return bool(np.isin(below, list(WIRE_POLE_IDS)).mean() > 0.03)


def is_direction_or_traffic_sign(bbox, img_hsv, img_bgr, seg_map, class_id,
                                  frame_h, frame_w, road_mask):
    """Route/place-name/tourist-direction boards and other traffic-authority
    signage get segmented as 'signboard'/'bulletin_board'/etc. just like
    commercial ads, but they're not sites anyone can rent. Two independent
    kinds of evidence have to agree before we reject one:
      1. COLOUR — dominated by one flat field colour + achromatic text,
         with very little other saturated content (works for green NH
         signs, blue signs, brown tourist/heritage signs, anything).
      2. STRUCTURE/POSITION — mounted overhead or roadside: elevated in
         the frame, AND (spans wide over the road, or has visible road
         passing directly underneath, or sits on a signpost, or is
         clipped at the frame edge the way pole/gantry signs often are).
    Requiring both cuts down on false positives against real flat-colour
    commercial hoardings, which are rarely elevated+over-the-road+pole-
    mounted all at once."""
    if class_id not in AD_IDS:
        return False, ""
    class_ids = {class_id}
    field_ratio, ach_ratio, other_ratio, field_sat = dominant_field_color_ratio(bbox, img_hsv, seg_map, class_ids)
    flat_uniform = (field_ratio + ach_ratio) > 0.68 and other_ratio < 0.24
    # Retroreflective road-sign paint (NH green, city blue, heritage brown)
    # is strongly saturated. A pale sky photo or soft product-photo
    # gradient can independently look "flat" (low hue variance) without
    # being real sign paint — this is exactly the failure mode that
    # misclassified a real-estate hoarding's sky-and-greenery creative
    # (pale blue field, ~85% flat/achromatic) as government signage.
    # Requiring genuine paint-level saturation on the field itself closes
    # that gap without weakening the check for actual road signs.
    field_is_painted = field_sat > 55
    if not flat_uniform or not field_is_painted:
        return False, ""

    x, y, w, h = bbox
    top_norm = y / frame_h
    wh_ratio = w / max(h, 1)
    elevated = top_norm < 0.50
    near_edge = (x <= 0.02 * frame_w) or (x + w >= frame_w * 0.98) or (y <= 0.02 * frame_h)
    wide_span = (w / frame_w) > 0.40 or wh_ratio > 3.0
    above_road = road_below_bbox_ratio(bbox, road_mask, frame_h, frame_w) > 0.12
    pole_mounted = pole_support_below(bbox, seg_map, frame_h)

    # Mere width/shape is NOT enough structural evidence on its own — a
    # wide, landscape-format board photographed up close (i.e. most
    # pedestrian shots of an ordinary building/shelter-mounted hoarding)
    # looks identical on width alone. Only genuine standalone-signpost
    # evidence (clipped at the frame edge, road visible directly beneath,
    # or a support pole beneath it) counts as structural proof; wide_span
    # can only reinforce one of those, not substitute for all of them.
    standalone_evidence = above_road or pole_mounted or near_edge
    if elevated and standalone_evidence:
        return True, (f"looks like a route/direction/place-name sign — flat single-colour "
                       f"field plus text ({(field_ratio+ach_ratio)*100:.0f}% flat/achromatic, "
                       f"minimal photographic content), mounted overhead or roadside "
                       f"— not a rentable ad placement")
    return False, ""


def is_clipped_overhead_signage(bbox, frame_w, frame_h, area_pct):
    """Catches small signage that's cut off at the frame edge and mounted
    high/overhead — the pattern typical of gantry route signs, tourist/
    heritage direction boards, and other pole-mounted informational
    signage (which come in every color, not just green/blue, so this is
    a color-independent complement to is_direction_or_traffic_sign).
    A surveyor also can't properly evaluate a board that's partially
    outside the frame, so this doubles as a 'can't fully assess it'
    filter regardless of what the sign turns out to be."""
    x, y, w, h = bbox
    margin_x = max(3, int(0.015 * frame_w))
    margin_y = max(3, int(0.015 * frame_h))
    touches_edge = (x <= margin_x) or (x + w >= frame_w - margin_x) or (y <= margin_y)
    elevated = (y / frame_h) < 0.35
    small = area_pct < 3.0
    return touches_edge and elevated and small


def hard_reject_reason(bbox, seg_map, img_bgr, img_hsv, frame_h, frame_w, area_pct, solidity, class_id, sky_mask, road_mask, auto_detected=False):
    x, y, w, h = bbox
    is_ad = class_id in AD_IDS
    is_building = class_id in BUILDING_IDS
    is_fence = class_id in FENCE_IDS
    is_wall = class_id in WALL_IDS
    cy_norm = (y + h/2) / frame_h
    top_norm = y / frame_h
    wh_ratio = w / max(h, 1)
    # NOTE: intentionally NOT filtering these checks down to only pixels
    # the segmentation model itself labeled with this surface's class.
    # Windows/doors/railings are their own separate ADE20K classes, so a
    # class-filtered mask would exclude exactly the pixels that prove a
    # surface is occupied/cluttered (see is_occupied_building_facade).
    mask_ids = None

    if area_pct < MIN_AREA_PCT and not is_ad:
        return f"too small ({area_pct*100:.1f}% < {MIN_AREA_PCT*100:.1f}%)"
    if area_pct < 0.003 and is_ad:
        return f"ad structure too small ({area_pct*100:.1f}%)"

    min_w = 50 if is_ad else MIN_W_PX
    min_h = 25 if is_ad else MIN_H_PX
    if w < min_w or h < min_h:
        if area_pct > MIN_AREA_PCT * 2:
            pass
        else:
            return f"too narrow/short ({w}×{h} px)"

    if wh_ratio < MIN_WH_RATIO:
        return f"too thin/tall (w/h={wh_ratio:.2f}) — pole or thin wall sliver"
    wh_max = MAX_WH_RATIO_AD if is_ad else MAX_WH_RATIO
    if wh_ratio > wh_max:
        return f"too horizontal (w/h={wh_ratio:.2f}) — strip, overpass, or road marking"

    if cy_norm < SKY_TOP_NORM:
        return f"in sky zone (centre at {cy_norm*100:.0f}% from top)"
    if cy_norm > GND_BOT_NORM:
        return f"at ground/road level (centre at {cy_norm*100:.0f}%)"

    sky_ov = sky_overlap_ratio(bbox, frame_h, seg_map)
    sky_limit = MAX_SKY_OVERLAP + 0.15 if solidity > 0.85 else MAX_SKY_OVERLAP
    if sky_ov > sky_limit:
        return f"overlaps sky zone ({sky_ov*100:.0f}% of bbox in sky)"

    if solidity < MIN_SOLIDITY and not is_ad:
        return f"fragmented/irregular mask (solidity={solidity:.2f} < {MIN_SOLIDITY})"

    is_fly, fly_reason = is_flyover_underside(bbox, seg_map, img_bgr, frame_h, frame_w, sky_mask, road_mask, class_id=class_id)
    if is_fly:
        return f"flyover/bridge underside — {fly_reason}"

    if wh_ratio > 4.0 and top_norm < 0.45 and not is_ad:
        return f"wide horizontal structure in upper frame — likely overpass (w/h={wh_ratio:.1f})"

    # Auto-detected "signboard"/"screen" sub-regions are our own heuristic
    # guess, not the model's semantic label — they get the stricter,
    # building-facade-grade glass threshold rather than the looser one
    # meant for genuine model-recognized ad structures.
    g_max = BUILDING_MAX_GLASS if (is_building or auto_detected) else GLASS_PCT_REJECT
    g_ratio = glass_ratio_in_box(bbox, seg_map, mask_ids)
    if g_ratio > g_max:
        return f"glass-heavy ({g_ratio*100:.0f}% windows/doors — not paintable)"

    occ_facade, occ_reason = is_occupied_building_facade(
        bbox, seg_map, img_bgr, is_building or auto_detected,
        is_genuine_building=is_building)
    if occ_facade:
        return occ_reason

    is_dir_sign, dir_reason = is_direction_or_traffic_sign(
        bbox, img_hsv, img_bgr, seg_map, class_id, frame_h, frame_w, road_mask)
    if is_dir_sign:
        return dir_reason

    if class_id in AD_IDS and is_clipped_overhead_signage(bbox, frame_w, frame_h, area_pct * 100):
        return ("clipped at the frame edge and mounted high/overhead — likely a gantry "
                 "or pole-mounted directional sign, and can't be fully assessed anyway")


    e_max = BUILDING_MAX_EDGE if (is_building or auto_detected) else EDGE_DENSITY_REJECT
    e_dens = edge_density_in_box(img_bgr, bbox, seg_map, mask_ids)
    if e_dens > e_max and (not is_ad or auto_detected):
        return f"cluttered surface (edge density={e_dens:.3f}) — windows/signage/AC units"

    occ = occluder_ratio_in_box(bbox, seg_map)
    if occ > OCC_PCT_REJECT:
        return f"heavily occluded ({occ*100:.0f}% of surface blocked by trees/vehicles)"

    if is_building:
        bot_norm = (y + h) / frame_h
        if bot_norm > BUILDING_MAX_HEIGHT_NORM:
            band_h = max(int(h * 0.25), 20)
            bottom_band = [x, y + h - band_h, w, band_h]
            d_ratio = door_ratio_in_box(bottom_band, seg_map)
            g_ratio2 = glass_ratio_in_box(bottom_band, seg_map)
            band_c_std = colour_uniformity(img_bgr, bottom_band)
            if d_ratio > 0.02 or g_ratio2 > 0.05 or band_c_std > COLOR_STD_REJECT:
                return f"building surface extends to ground-floor level ({bot_norm*100:.0f}%) — shopfront, not hoarding"

    if is_fence and area_pct < 0.025:
        return f"fence too small ({area_pct*100:.1f}%) — not useful for advertising"

    return None


# ═════════════════════════════════════════════════════════════════════════
# Candidate extraction
# ═════════════════════════════════════════════════════════════════════════

MERGE_OCCLUDER_IDS = TREE_IDS | WIRE_POLE_IDS | LIGHT_IDS


def _boxes_are_same_panel(b1, b2, frame_w, frame_h, seg_map):
    """True if two ad-class boxes look like pieces of one physical panel
    rather than two separate structures.

    Two independent conditions can trigger a merge:
      1. Genuine overlap — the segmentation model itself labeled the same
         physical region with two different ad-related classes (e.g. the
         top of one poster came out "poster", the bottom "signboard"),
         so the two pieces were extracted in different passes over
         ALL_SURFACE_IDS and never got IoU-deduped against each other.
      2. A THIN gap that is itself dominated by an occluder class
         (pole/wire/tree branch) — i.e. something physically crossed in
         front of one continuous surface when the photo was taken.

    Deliberately NOT included: merely being close together. A gap that's
    small in pixels but shows open wall/sky/background (not an occluder)
    is real physical separation between two different structures, and
    the gap is capped at a small ABSOLUTE size (not a fraction of frame
    width) so this can't daisy-chain across a whole street scene the way
    a blanket morphological-closing bridge could."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
    y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)

    # Condition 1: substantial overlap on both axes = same region, different class label.
    if x_overlap > 0.4 * min(w1, w2) and y_overlap > 0.4 * min(h1, h2):
        return True

    x_gap = max(x1, x2) - min(x1 + w1, x2 + w2)
    y_gap = max(y1, y2) - min(y1 + h1, y2 + h2)
    # A pole/wire/branch is typically a few percent of the surface's own
    # size, not a few percent of the whole frame — capped in absolute
    # pixels so it can't bridge real, larger gaps between separate
    # structures on a wide/zoomed-out shot.
    max_gap_px = min(45, max(12, int(0.01 * frame_w)))

    def gap_is_occluder(gx1, gy1, gx2, gy2):
        gx1, gy1 = max(0, int(gx1)), max(0, int(gy1))
        gx2, gy2 = min(frame_w, int(gx2)), min(frame_h, int(gy2))
        if gx2 <= gx1 or gy2 <= gy1:
            return False
        strip = seg_map[gy1:gy2, gx1:gx2]
        if strip.size == 0:
            return False
        return float(np.isin(strip, list(MERGE_OCCLUDER_IDS)).mean()) > 0.35

    # Condition 2a: stacked one above the other, thin occluder-filled gap between.
    if x_overlap > 0.5 * min(w1, w2) and 0 <= y_gap <= max_gap_px:
        gx1, gx2 = max(x1, x2), min(x1 + w1, x2 + w2)
        gy1, gy2 = min(y1 + h1, y2 + h2), max(y1, y2)
        if gap_is_occluder(gx1, gy1, gx2, gy2):
            return True

    # Condition 2b: side by side, thin occluder-filled gap between.
    if y_overlap > 0.5 * min(h1, h2) and 0 <= x_gap <= max_gap_px:
        gy1, gy2 = max(y1, y2), min(y1 + h1, y2 + h2)
        gx1, gx2 = min(x1 + w1, x2 + w2), max(x1, x2)
        if gap_is_occluder(gx1, gy1, gx2, gy2):
            return True

    return False


def _merge_adjacent_ad_panels(results, frame_w, frame_h, seg_map, img_bgr, img_hsv, sky_mask, road_mask):
    """Post-pass: unions touching/overlapping AD_IDS candidates that look
    like one physical panel (see _boxes_are_same_panel) into a single
    candidate, recomputes its hard-reject reason, and removes the pieces
    it replaced. Runs in place on `results`.

    A merged group is only accepted if its combined bbox is still a
    plausible single surface (bounded area, not a sprawling span) —
    protects against a chain of individually-small merges (A merges with
    B, B merges with C, ...) adding up to something covering a large
    fraction of the frame, which is real, separate structures being
    fused rather than one occluded surface."""
    idxs = [i for i, r in enumerate(results) if r["is_ad"]]
    if len(idxs) < 2:
        return
    parent = {i: i for i in idxs}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            i, j = idxs[a], idxs[b]
            if _boxes_are_same_panel(results[i]["bbox"], results[j]["bbox"], frame_w, frame_h, seg_map):
                union(i, j)

    groups = {}
    for i in idxs:
        groups.setdefault(find(i), []).append(i)

    to_remove = set()
    to_add = []
    MAX_MERGED_AREA_PCT = 0.18
    for members in groups.values():
        if len(members) < 2:
            continue
        xs1 = [results[i]["bbox"][0] for i in members]
        ys1 = [results[i]["bbox"][1] for i in members]
        xs2 = [results[i]["bbox"][0] + results[i]["bbox"][2] for i in members]
        ys2 = [results[i]["bbox"][1] + results[i]["bbox"][3] for i in members]
        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        area_px = sum(results[i]["area_px"] for i in members)
        area_pct = area_px / (frame_w * frame_h)
        if area_pct > MAX_MERGED_AREA_PCT:
            # This group grew too large to plausibly be one physical
            # surface — leave the original pieces as separate candidates
            # rather than fusing them.
            continue
        solidity = min(results[i]["solidity"] for i in members)
        biggest = max(members, key=lambda i: results[i]["area_px"])
        cid, cname = results[biggest]["class_id"], results[biggest]["class_name"]
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        reason = hard_reject_reason(bbox, seg_map, img_bgr, img_hsv, frame_h, frame_w,
                                     area_pct, solidity, cid, sky_mask, road_mask, auto_detected=False)
        to_add.append(dict(
            class_id=cid, class_name=cname, is_ad=True, is_building=False,
            is_wall=False, is_fence=False, auto_detected=False,
            poly=poly, bbox=bbox, area_px=area_px,
            area_pct=round(area_pct * 100, 2), solidity=solidity,
            split=False, hard_reject=reason,
        ))
        to_remove.update(members)

    if to_remove:
        for i in sorted(to_remove, reverse=True):
            del results[i]
        results.extend(to_add)


def extract_candidates(seg_map, img_bgr, img_hsv, frame_h, frame_w, sky_mask, road_mask):
    fa = frame_h * frame_w
    results = []
    seen = []

    def is_dup(b):
        x, y, w, h = b
        for bx, by, bw, bh in seen:
            ix1 = max(x, bx); iy1 = max(y, by)
            ix2 = min(x+w, bx+bw); iy2 = min(y+h, by+bh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = w*h + bw*bh - inter
                if union > 0 and inter/union > 0.45:
                    return True
        return False

    def add_candidate(cid, cname, poly, bbox, area_px, solidity, split=False, auto_detected=False):
        if bbox is None:
            return False
        if is_dup(bbox):
            return False
        area_pct = area_px / fa
        reason = hard_reject_reason(bbox, seg_map, img_bgr, img_hsv, frame_h, frame_w, area_pct, solidity, cid, sky_mask, road_mask, auto_detected=auto_detected)
        results.append(dict(
            class_id=cid, class_name=cname,
            is_ad=cid in AD_IDS, is_building=cid in BUILDING_IDS,
            is_wall=cid in WALL_IDS, is_fence=cid in FENCE_IDS,
            auto_detected=auto_detected,
            poly=poly, bbox=bbox, area_px=area_px,
            area_pct=round(area_pct * 100, 2), solidity=solidity,
            split=split, hard_reject=reason,
        ))
        seen.append(bbox)
        return True

    for cid in ALL_SURFACE_IDS:
        if (seg_map == cid).sum() == 0:
            continue
        cname = CLASS_PALETTE.get(cid, (f"class_{cid}", (200, 200, 200)))[0]
        raw = (seg_map == cid).astype(np.uint8)
        cleaned = remove_foreground(raw, seg_map)
        cleaned = morph_clean(cleaned, close=11 if cid in BUILDING_IDS else 5)
        n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(cleaned)

        if cid in BUILDING_IDS:
            for comp_id in range(1, n_lbl):
                sx, sy, sw, sh, s_area = stats[comp_id]
                if s_area / fa < 0.01:
                    continue
                sub_bbox = (sx, sy, sw, sh)
                comp_mask = (lbl_map == comp_id).astype(np.uint8)
                screens = detect_screen_subregions(comp_mask, sub_bbox, img_bgr, img_hsv)
                dark_signs = detect_dark_signage_subregions(comp_mask, sub_bbox, img_bgr, img_hsv)
                for (bx, by, bw, bh) in dark_signs:
                    dpoly = [[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]]
                    add_candidate(43, "signboard (auto-detected)", dpoly, [bx, by, bw, bh], bw*bh, solidity=0.9, auto_detected=True)
                for (bx, by, bw, bh) in screens:
                    poly = [[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]]
                    area_px = bw * bh
                    add_candidate(130, "screen (auto-detected)", poly, [bx, by, bw, bh], area_px, solidity=0.9, auto_detected=True)

        for i in range(1, n_lbl):
            sx, sy, sw, sh, s_area = stats[i]
            min_area_frac = 0.0012 if cid in AD_IDS else 0.003
            if s_area / fa < min_area_frac:
                continue
            comp = (lbl_map == i).astype(np.uint8)

            if cid in (BUILDING_IDS | AD_IDS) and s_area / fa > SPLIT_AT_PCT:
                split_class_ids = BUILDING_IDS if cid in BUILDING_IDS else AD_IDS
                is_ad_split = cid in AD_IDS
                whole_poly, whole_bbox, whole_area, whole_sol = tight_rect(comp)
                needs_split = False
                if whole_bbox is not None:
                    if is_ad_split:
                        # For an existing ad structure, printed/photographic
                        # creative content is ALWAYS visually busy — bright
                        # colour blocks, product photography, text, logos —
                        # regardless of whether it's one panel or several.
                        # Using edge-density/colour-variance/glass here (as
                        # the building branch does) would flag almost every
                        # real single-panel ad as "needs splitting" purely
                        # because its creative is colourful, which is what
                        # was fragmenting a single KitKat-style poster into
                        # multiple recommended sites. Only a genuine
                        # structural gap in the surface's own mask — found
                        # via the same wide-min-gap probe used for the
                        # actual split below — counts as evidence that this
                        # is more than one physical panel.
                        probe_gap = max(30, int(0.02 * (frame_w if frame_w else sw)))
                        needs_split = bool(natural_row_splits(comp, sy, sh, min_gap=probe_gap) or
                                           natural_col_splits(comp, sx, sw, min_gap=probe_gap))
                    else:
                        e_dens_whole = edge_density_in_box(img_bgr, whole_bbox, seg_map, None)
                        d_ratio_whole = door_ratio_in_box(whole_bbox, seg_map, None)
                        g_ratio_whole = glass_ratio_in_box(whole_bbox, seg_map, None)
                        c_std_whole = colour_uniformity(img_bgr, whole_bbox, seg_map, None)
                        needs_split = (e_dens_whole > SPLIT_EDGE_DENSITY_THRESH or
                                       d_ratio_whole > SPLIT_DOOR_THRESH or
                                       g_ratio_whole > SPLIT_GLASS_THRESH or
                                       c_std_whole > COLOR_STD_REJECT)
                if not needs_split:
                    add_candidate(cid, cname, whole_poly, whole_bbox, whole_area, whole_sol)
                    continue

                # An existing ad structure is very often printed/mounted as
                # several colour panels side by side (a pink promo strip
                # next to a yellow route-number strip, a building photo
                # next to a plain QR-code panel...) with no real physical
                # gap between them — that's one rentable hoarding face, not
                # several. A thin support bracket or shadow line between
                # panels can still register as a few near-empty pixel
                # columns, so a small min_gap (fine for finding genuine
                # separations between distinct BUILDING sections) makes
                # AD components fragment along every internal panel seam.
                # Requiring a much wider gap means only a real structural
                # break — actual physical distance between two separate
                # hoardings — triggers a split.
                gap = max(30, int(0.02 * (frame_w if frame_w else sw))) if is_ad_split else 6
                row_splits = natural_row_splits(comp, sy, sh, min_gap=gap)
                row_cuts = [sy] + row_splits + [sy+sh]
                any_zone_added = False
                for r in range(len(row_cuts)-1):
                    zy, zy2 = row_cuts[r], row_cuts[r+1]
                    if zy2 - zy < MIN_H_PX:
                        continue
                    row_zone = comp.copy()
                    row_zone[:zy, :] = 0; row_zone[zy2:, :] = 0
                    if row_zone.sum() == 0:
                        continue
                    col_splits = natural_col_splits(row_zone, sx, sw, min_gap=gap)
                    if col_splits:
                        col_cuts = [sx] + col_splits + [sx+sw]
                        pieces = []
                        for j in range(len(col_cuts)-1):
                            zx, zx2 = col_cuts[j], col_cuts[j+1]
                            if zx2 - zx < MIN_W_PX:
                                continue
                            zm = row_zone.copy()
                            zm[:, :zx] = 0; zm[:, zx2:] = 0
                            _, zbbox, _, _ = tight_rect(zm)
                            if zbbox is not None:
                                pieces.append((zm, zbbox))
                    else:
                        pieces = [(row_zone, [sx, zy, sw, zy2-zy])]

                    leaves = []
                    if is_ad_split:
                        # Gap-based splitting above already isolated any
                        # genuinely separate structures. Don't further
                        # slice each piece by internal colour boundaries —
                        # that's normal ad-creative design, not evidence
                        # of a different rentable surface.
                        leaves = pieces
                    else:
                        for zm, zbbox in pieces:
                            recursive_colour_split(zm, zbbox, seg_map, img_bgr, split_class_ids, out=leaves)

                    for zm, zbbox in leaves:
                        poly, bbox, ca, sol = tight_rect(zm)
                        if add_candidate(cid, cname, poly, bbox, ca, sol, split=True):
                            any_zone_added = True

                if not any_zone_added and whole_bbox is not None:
                    add_candidate(cid, cname, whole_poly, whole_bbox, whole_area, whole_sol)
                continue

            poly, bbox, ca, sol = tight_rect(comp)
            add_candidate(cid, cname, poly, bbox, ca, sol)

    _merge_adjacent_ad_panels(results, frame_w, frame_h, seg_map, img_bgr, img_hsv, sky_mask, road_mask)
    return results


# ═════════════════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════════════════

def score_candidate(c, seg_map, img_bgr, frame_h, frame_w):
    bbox = c["bbox"]
    is_ad = c["is_ad"]
    is_bldg = c["is_building"]
    auto_detected = c.get("auto_detected", False)
    area_pct = c["area_pct"] / 100.0

    occ_raw = occluder_ratio_in_box(bbox, seg_map)
    s_occ = round(max(0.0, 1.0 - occ_raw * 4.0), 3)

    s_pos = vertical_position_score(bbox, frame_h)

    if is_ad and not auto_detected:
        # A genuine model-recognized ad class (segmentation actually
        # labeled this "signboard"/"poster"/etc.) gets the confident
        # "existing infrastructure" score.
        s_surface = round(min(1.0, 0.85 + (1.0 - occ_raw) * 0.15), 3)
    else:
        mask_ids = BUILDING_IDS if is_bldg else None
        s_surface = blank_wall_score(img_bgr, bbox, seg_map if is_bldg else None, None)
        g_ratio = glass_ratio_in_box(bbox, seg_map)
        s_surface = round(max(0.0, s_surface - g_ratio * 2.0), 3)
        if is_bldg:
            s_surface = round(s_surface * 0.85, 3)
        if auto_detected:
            # Our own contrast/colour heuristic guessed this was signage —
            # it's not a confirmed flat printed surface, so it shouldn't
            # score as confidently as one even after passing the gate.
            s_surface = round(s_surface * 0.9, 3)

    s_size = size_score(area_pct, is_ad)
    s_shape = shape_score(bbox)
    ad_mult = 1.08 if (is_ad and not auto_detected) else 1.0

    raw = (W_OCC * s_occ + W_POS * s_pos + W_SURFACE * s_surface + W_SIZE * s_size + W_SHAPE * s_shape)
    # Cap below a perfect 1.00 — even an excellent site has some inherent
    # uncertainty (viewing angle, lighting, wear), so a "flawless" score
    # would look unrealistic. MAX_REALISTIC_SCORE leaves headroom.
    score = round(min(MAX_REALISTIC_SCORE, raw * ad_mult), 4)

    if is_bldg and score >= THR_PREMIUM:
        # A building facade is capped at VIABLE, never PREMIUM, regardless
        # of how blank/clean it measures. Unlike an existing signboard/
        # poster structure (which already has installed infrastructure and
        # likely legal clearance), a bare building elevation always needs
        # a site visit: owner/RWA no-objection, a structural safety
        # certificate for anything above a modest size, and confirmation
        # it isn't a residential facade — none of which a photo can prove.
        score = BUILDING_SCORE_CAP

    if is_ad and area_pct < 0.01:
        score = min(score, THR_PREMIUM - 0.02)

    # A tiny ad-class detection that also sits at a poor pedestrian
    # eye-line (very low or very high in frame) is more likely a stray
    # or mislabeled region — a distant plaque, meter box, sticker — than
    # a genuinely usable placement, even though the "existing ad
    # structure" confidence bonus alone could otherwise push it over the
    # viable line on size/occlusion/shape scores.
    if is_ad and area_pct < 0.012 and s_pos < 0.35:
        score = min(score, THR_VIABLE - 0.02)

    if score >= THR_PREMIUM:
        rec = "PREMIUM"
    elif score >= THR_VIABLE:
        rec = "VIABLE"
    else:
        rec = "AVOID"

    x, y, w, h = bbox
    cx_n = (x + w/2) / frame_w
    h_zone = ("Far-Left" if cx_n < 0.15 else
              "Left" if cx_n < 0.38 else
              "Near-Center" if cx_n < 0.62 else
              "Right" if cx_n < 0.85 else "Far-Right")
    size_l = ("Large" if area_pct >= 0.07 else "Medium" if area_pct >= 0.022 else "Small")
    occ_l = ("Clear" if s_occ >= 0.75 else "Partial" if s_occ >= 0.40 else "Heavy")

    bs_display = blank_wall_score(img_bgr, bbox) if (not is_ad or auto_detected) else 1.0

    c.update(dict(
        score=score, recommendation=rec,
        s_occ=s_occ, s_pos=s_pos, s_surface=s_surface,
        s_size=s_size, s_shape=s_shape,
        occ_raw=round(occ_raw, 3),
        glass_raw=round(glass_ratio_in_box(bbox, seg_map), 3),
        edge_density=round(edge_density_in_box(img_bgr, bbox), 3),
        blank_score=bs_display,
        h_zone=h_zone, size_label=size_l, occ_label=occ_l,
    ))
    return c


# ═════════════════════════════════════════════════════════════════════════
# Drawing
# ═════════════════════════════════════════════════════════════════════════

def _outline_points(s):
    """Returns an (N,2) int32 array to draw for this site — the clean
    silhouette polygon from segmentation when available, otherwise the
    bounding box rectangle as a fallback."""
    poly = s.get("poly")
    if poly and len(poly) >= 3:
        pts = np.array(poly, dtype=np.int32)
        if pts.shape[0] >= 3:
            return pts
    x, y, w, h = [int(v) for v in s["bbox"]]
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)


def draw_results(img_rgb, scored, frame_h, frame_w):
    out = img_rgb.copy().astype(np.float32)
    display = [s for s in scored if s["recommendation"] != "AVOID"]

    if not display:
        out = img_rgb.copy()
        cv2.rectangle(out, (12, 12), (frame_w-12, 72), (15, 15, 15), -1)
        cv2.putText(out, "NO SUITABLE BILLBOARD SURFACES DETECTED",
                    (26, 48), cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 50, 50), 2)
        return out

    # Fill sites by rasterizing each polygon into its own mask rather than
    # tinting a full bounding rectangle — this is what keeps the highlight
    # hugging the actual wall/surface shape instead of a wide box.
    for s in display:
        pts = _outline_points(s)
        col = REC_COL[s["recommendation"]]
        alpha = 0.16 if s["recommendation"] == "PREMIUM" else 0.09
        layer = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(layer, [pts], 1)
        idx = layer.astype(bool)
        fill_col = np.array(col, dtype=np.float32)
        out[idx] = out[idx] * (1 - alpha) + fill_col * alpha

    out = np.clip(out, 0, 255).astype(np.uint8)

    for s in display:
        pts = _outline_points(s)
        col = REC_COL[s["recommendation"]]
        lw = 3 if s["recommendation"] == "PREMIUM" else 2
        cv2.polylines(out, [pts], isClosed=True, color=col, thickness=lw, lineType=cv2.LINE_AA)

        x_top, y_top = int(pts[:, 0].min()), int(pts[:, 1].min())
        label = f"{s.get('sid','')}  {s['recommendation']}  {s['score']:.2f}"
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        ly = max(th + 8, y_top - 3)
        cv2.rectangle(out, (x_top, ly - th - 8), (x_top + tw + 10, ly), (10, 10, 10), -1)
        cv2.putText(out, label, (x_top + 5, ly - 5), font, fs, col, ft, cv2.LINE_AA)

    return out


# ═════════════════════════════════════════════════════════════════════════
# "Why recommended" bullets (single-image version — no video/dwell data)
# ═════════════════════════════════════════════════════════════════════════

def build_bullets(s):
    bullets = []
    bullets.append(
        f"Detected as {s['class_name']}, {s['h_zone']} of frame, "
        f"{s['size_label'].lower()} surface (~{s['area_pct']:.1f}% of frame area)"
    )
    if s["is_ad"] and not s.get("auto_detected"):
        bullets.append("Existing ad structure — already has installed infrastructure and likely legal clearance")
    else:
        bullets.append(f"Blank/flat surface quality: {s['blank_score']*100:.0f}%")

    bullets.append(f"{s['occ_label']} sightline for a pedestrian (occlusion score {s['s_occ']:.2f})")
    bullets.append(f"Pedestrian eye-level positioning score: {s['s_pos']:.2f}")
    bullets.append(f"Visual clutter around the surface (edge density): {s['edge_density']:.2f}")
    if s.get("glass_raw", 0) > 0.01 and (not s["is_ad"] or s.get("auto_detected")):
        bullets.append(f"Glass/window content: {s['glass_raw']*100:.0f}% of surface")

    # India-specific OOH compliance notes — a photo can only judge what's
    # visually there, not who owns the wall or what a municipality allows,
    # so every recommendation carries the relevant real-world caveat.
    if s["is_building"]:
        bullets.append(
            "⚠ Full building facade — requires owner/RWA no-objection and a "
            "structural safety certificate before mounting anything sizeable; "
            "most Indian municipal outdoor-ad byelaws (e.g. Delhi's Outdoor "
            "Advertising Policy, BMC/MCGM, GHMC) treat this differently from a "
            "compound wall or a pre-licensed hoarding, and don't recommend "
            "advertising on lived-in residential elevations at all"
        )
    elif s["is_wall"] and not s.get("auto_detected"):
        bullets.append(
            "Compound/boundary wall — still needs the property owner's consent "
            "and, in most cities, a permit from the local municipal "
            "corporation before installation"
        )
    elif s["is_ad"] and not s.get("auto_detected"):
        bullets.append(
            "Existing licensed structure — verify the current permit is valid "
            "and unexpired before booking, rather than assuming clearance carries over"
        )
    return bullets


# ═════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════

def analyze_image(image_path, annotated_out_path):
    """
    Runs the full pipeline on a single image.
    Returns a dict with: frame_w, frame_h, scored (all sites, ranked),
    rejected (hard-rejected candidates), avoid (passed gate but low score),
    annotated_image_path.
    """
    img_bgr_raw = cv2.imread(image_path)
    if img_bgr_raw is None:
        raise ValueError("Could not read image file — unsupported format or corrupt file.")

    orig_h, orig_w = img_bgr_raw.shape[:2]

    # Upscale low-resolution uploads before doing anything else. See
    # ANALYSIS_MIN_WIDTH above — this keeps every absolute-pixel heuristic
    # (candidate size floors, morphology kernels, sub-region detectors)
    # behaving the same regardless of how small the source photo was.
    if orig_w < ANALYSIS_MIN_WIDTH:
        scale = ANALYSIS_MIN_WIDTH / orig_w
        new_w = ANALYSIS_MIN_WIDTH
        new_h = int(round(orig_h * scale))
        img_bgr_raw = cv2.resize(img_bgr_raw, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    frame_h, frame_w = img_rgb.shape[:2]

    seg_map = segment(Image.fromarray(img_rgb))
    sky_mask = np.isin(seg_map, list(SKY_IDS)).astype(np.uint8)
    road_mask = np.isin(seg_map, list(ROAD_IDS)).astype(np.uint8)

    all_candidates = extract_candidates(seg_map, img_bgr, img_hsv, frame_h, frame_w, sky_mask, road_mask)
    passed = [c for c in all_candidates if c["hard_reject"] is None]
    rejected = [c for c in all_candidates if c["hard_reject"] is not None]

    scored = [score_candidate(c, seg_map, img_bgr, frame_h, frame_w) for c in passed]

    ad_scored = [s for s in scored if s["is_ad"]]
    if ad_scored:
        max_ad_area = max(s["area_pct"] for s in ad_scored)
        for s in ad_scored:
            if max_ad_area > 0 and s["area_pct"] < max_ad_area * 0.5:
                s["score"] = min(s["score"], THR_PREMIUM - 0.02)
                s["recommendation"] = ("PREMIUM" if s["score"] >= THR_PREMIUM else
                                        "VIABLE" if s["score"] >= THR_VIABLE else "AVOID")

    scored.sort(key=lambda s: s["score"], reverse=True)

    display = [s for s in scored if s["recommendation"] != "AVOID"]
    for i, s in enumerate(display, 1):
        s["sid"] = f"SITE-{i:03d}"
        s["bullets"] = build_bullets(s)

    avoid_s = [s for s in scored if s["recommendation"] == "AVOID"]

    result_img = draw_results(img_rgb, scored, frame_h, frame_w)

    # If this photo was upscaled for analysis, scale the annotated result
    # back down to the resolution the person actually uploaded, so what
    # they see matches their original file rather than an enlarged copy.
    if (frame_w, frame_h) != (orig_w, orig_h):
        result_img = cv2.resize(result_img, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(annotated_out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    return dict(
        frame_w=orig_w, frame_h=orig_h,
        sites=display,
        rejected=rejected,
        avoid=avoid_s,
        annotated_image_path=annotated_out_path,
    )
