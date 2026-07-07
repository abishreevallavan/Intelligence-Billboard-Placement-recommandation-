"""
engine_driver_video.py — Video extension of the driver-POV engine.

Takes a short driver-POV video (dashcam / phone-mounted), samples it at
~1 frame per second, and runs the *same* per-frame detection + scoring
pipeline as engine_driver.py (segmentation -> candidate extraction ->
hard-reject gate -> scoring) on every sampled frame. Nothing about the
per-frame scoring logic is changed or duplicated — it's imported and
reused as-is from engine_driver.py.

What's added on top, specifically for video:
  1. Frame sampling (1 fps, capped at MAX_SECONDS) via OpenCV.
  2. A lightweight cross-frame tracker: the same physical billboard/wall
     will appear in several consecutive sampled frames (usually at a
     different position/size as the vehicle approaches or passes it).
     Frames' candidates are linked into "tracks" using IoU + a
     center-distance/area-ratio fallback (bboxes shift a lot at 1 fps).
  3. Dwell score: fraction of sampled frames a track was detected in.
     This is the one new factor requested — how much of the drive-by
     window the surface was actually visible/trackable for — and it is
     blended into the final ranking score alongside the existing
     per-frame visual score (occlusion/position/surface/size/shape).
  4. Top 3-5 sites are returned, each with its single best-scoring
     frame annotated (clearest/largest view of that surface), so the
     UI can show one representative image per recommended site instead
     of 20 near-duplicate frames.

Performance notes (kept deliberately simple/cheap):
  - The SegFormer model is loaded once (engine_driver's module-level
    cache) and reused across all sampled frames — it is not reloaded
    per frame.
  - Segmentation runs in small batches (BATCH_SIZE frames per forward
    pass) instead of one-by-one, cutting per-call Python/tensor-setup
    overhead.
  - Only 1 frame/sec is analyzed (not every video frame), so a 20s clip
    means at most 20 segmentation passes total, regardless of the
    source video's actual frame rate.
"""

import os
import math
import numpy as np
import cv2
from PIL import Image

import engine_driver as ED

# ── Config ───────────────────────────────────────────────────────────────
MAX_SECONDS   = 20     # hard cap on how much of the video we analyze
TARGET_FPS    = 1      # 1 sampled frame per second
BATCH_SIZE    = 4      # frames per segmentation forward pass

IOU_MATCH_MIN      = 0.08   # minimum IoU to link a candidate to a track
CENTER_DIST_NORM   = 0.16   # or: center within this fraction of frame width...
                             # ...per second of gap since the track was last seen
AREA_RATIO_LO      = 0.30   # ...and area didn't change more than this much
AREA_RATIO_HI      = 3.50
MAX_FRAME_GAP      = 3      # allow a track to "disappear" for up to N sampled frames

VEHICLE_RING_MARGIN     = 0.25   # expand bbox by this fraction to build the "ring" checked for vehicle pixels
VEHICLE_RING_REJECT     = 0.15   # ring vehicle-pixel overlap above this -> likely a vehicle body panel
ROAD_IN_BOX_REJECT      = 0.30   # candidate box itself sitting mostly on road pixels -> likely a vehicle, not a fixed surface

DWELL_WEIGHT  = 0.15   # how much dwell (occurrence frequency) contributes
                        # to the final ranking score, on top of visual score

# Multiplicative penalty applied to any track only seen in a single
# sampled frame (see finalize_tracks) — but only once it's already
# cleared UNCONFIRMED_MIN_BASE_SCORE below. This is deliberately mild:
# it's a small tie-break in favor of multi-frame-confirmed tracks, not
# a mechanism for rejecting weak detections (that's what the min-score
# gate is for).
LOW_CONFIDENCE_PENALTY = 0.92

# A single-frame (unconfirmed) track needs at least this much raw
# visual quality — occlusion, sightline, size, position, all already
# baked into base_score — to be shown at all. Below this, a one-off
# sighting is treated as likely segmentation noise and forced to
# AVOID. At/above this, it's treated as a real surface the vehicle
# simply passed too quickly to re-confirm, and only gets the mild
# LOW_CONFIDENCE_PENALTY above rather than being discarded.
UNCONFIRMED_MIN_BASE_SCORE = 0.62

TOP_MAX       = 5

# Broad surface categories used for cross-frame matching. Per-frame semantic
# segmentation is noisy: the *same* physical wall can flip between "wall",
# "building", and "fence" labels from one sampled second to the next. Matching
# on exact class_name caused tracks to break almost every frame in practice
# (a surface would show up as "seen in 1 of 13 frames" even when it was
# visible the whole clip). Grouping into broad categories and matching within
# a group — combined with spatial proximity — makes tracking far more robust
# to that per-frame label flicker.
_SURFACE_GROUPS = {
    "signboard": "ad", "poster": "ad", "trade_name": "ad", "screen": "ad", "bulletin_board": "ad",
    "wall": "wall", "pillar": "wall", "building": "wall", "house": "wall", "skyscraper": "wall",
    "fence": "fence", "railing": "fence", "awning": "fence", "booth": "fence", "canopy": "fence",
}


def _surface_group(class_name):
    base = class_name.replace(" (auto-detected)", "").strip()
    return _SURFACE_GROUPS.get(base, base)


# ═════════════════════════════════════════════════════════════════════════
# 1. Frame extraction
# ═════════════════════════════════════════════════════════════════════════

def extract_frames(video_path, tmp_dir, target_fps=TARGET_FPS, max_seconds=MAX_SECONDS):
    """Pulls ~1 frame/sec from the video, up to max_seconds. Returns a
    list of dicts: idx, ts (seconds), bgr, rgb, hsv, path (jpg on disk,
    needed because engine_driver.segment() reads from a file path).

    NOTE: frames are pulled via sequential decode (cap.read() in a loop),
    NOT via cap.set(CAP_PROP_POS_FRAMES, ...) random-access seeking.
    Seeking-by-frame-number is unreliable on many MP4/H.264 files — the
    backend often snaps to the nearest preceding keyframe instead of the
    exact requested frame, silently returning a frame from a different
    timestamp than asked for. Since the entire cross-frame tracker below
    depends on box position/size changing *smoothly* between sampled
    seconds, even occasional mis-seeks corrupt that continuity and were
    a major cause of tracks breaking/merging incorrectly. Sequential
    decode always returns the true next frame, at the cost of decoding
    (not analyzing) the in-between frames we skip."""
    os.makedirs(tmp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file — unsupported format or corrupt file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / video_fps) if (video_fps > 0 and total_frames > 0) else max_seconds

    n_seconds = int(min(max_seconds, math.floor(duration))) if duration > 0 else max_seconds
    n_seconds = max(n_seconds, 1)
    step = 1.0 / target_fps

    # Precompute which raw frame numbers we want, then decode sequentially
    # and only keep the frame once we reach/pass each target frame number.
    targets = []
    t = 0.0
    while t < n_seconds:
        targets.append(t)
        t += step
    target_frame_numbers = [int(round(tt * video_fps)) for tt in targets]

    frames = []
    raw_idx = -1
    ti = 0
    while ti < len(target_frame_numbers):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        raw_idx += 1
        if raw_idx < target_frame_numbers[ti]:
            continue
        idx = len(frames)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        path = os.path.join(tmp_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(path, frame_bgr)
        frames.append(dict(idx=idx, ts=round(targets[ti], 2), bgr=frame_bgr, rgb=rgb, hsv=hsv, path=path))
        ti += 1
        while ti < len(target_frame_numbers) and target_frame_numbers[ti] <= raw_idx:
            ti += 1
    cap.release()
    return frames


# ═════════════════════════════════════════════════════════════════════════
# 2. Batched segmentation (reuses the already-loaded model from engine_driver)
# ═════════════════════════════════════════════════════════════════════════

def segment_batch(frame_paths, batch_size=BATCH_SIZE):
    """Same output as engine_driver.segment() per image, but runs frames
    through the model in small batches to cut overhead. Returns a list
    of seg_map arrays, one per input path, in order."""
    import torch
    import torch.nn.functional as F

    ED._load_model()
    proc, model, device = ED._processor, ED._model, ED._device

    seg_maps = []
    for i in range(0, len(frame_paths), batch_size):
        chunk_paths = frame_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk_paths]
        sizes = [im.size for im in imgs]  # (W, H) per image
        inputs = proc(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        for j, (W, H) in enumerate(sizes):
            seg = (F.interpolate(logits[j:j + 1], (H, W), mode="bilinear", align_corners=False)
                     .argmax(1).squeeze().cpu().numpy().astype(np.int32))
            seg_maps.append(seg)
    return seg_maps


# ═════════════════════════════════════════════════════════════════════════
# 3. Vehicle-misclassification guard
# ═════════════════════════════════════════════════════════════════════════

def _vehicle_context_reject(bbox, seg_map, road_mask, frame_w, frame_h):
    """Segmentation sometimes mislabels a vehicle's flat side panel as
    "wall"/"fence" instead of "vehicle" — there's no vehicle-class pixel
    inside that box for the existing occlusion check to catch, since the
    mislabel *is* the problem. Two cheap tells catch most real cases:

      1. The box itself sits mostly on road-classified pixels — a fixed
         wall/fence/signboard is set back from the road surface, but a
         vehicle body IS on the road.
      2. A ring just outside the box has meaningful vehicle-classified
         pixels — e.g. the wheels, mirrors, or windows of the same
         vehicle that weren't misclassified, sitting right next to the
         "wall-like" panel.

    Returns True if the candidate looks like it's actually part of a
    vehicle rather than a fixed surface.
    """
    x, y, w, h = [int(v) for v in bbox]
    roi_road = road_mask[y:y+h, x:x+w]
    if roi_road.size > 0 and float(roi_road.mean()) > ROAD_IN_BOX_REJECT:
        return True

    mx = int(w * VEHICLE_RING_MARGIN)
    my = int(h * VEHICLE_RING_MARGIN)
    rx1, ry1 = max(0, x - mx), max(0, y - my)
    rx2, ry2 = min(frame_w, x + w + mx), min(frame_h, y + h + my)
    ring = seg_map[ry1:ry2, rx1:rx2].copy()
    # exclude the interior (the candidate box itself) so we only look at
    # the surrounding ring, not the box's own (mislabeled) pixels
    iy1, iy2 = max(0, y - ry1), max(0, y - ry1 + h)
    ix1, ix2 = max(0, x - rx1), max(0, x - rx1 + w)
    interior_mask = np.zeros(ring.shape, dtype=bool)
    interior_mask[iy1:iy2, ix1:ix2] = True
    ring_only = ring[~interior_mask]
    if ring_only.size == 0:
        return False
    vehicle_ratio = float(np.isin(ring_only, list(ED.VEHICLE_IDS)).sum()) / ring_only.size
    if vehicle_ratio > VEHICLE_RING_REJECT:
        return True

    # A van/bus parked on a shoulder or footpath (not "road"-classified
    # pixels) can still butt directly up against the candidate box on
    # one side rather than surrounding it — the symmetric "ring" check
    # above can dilute that down below threshold. Also check immediately
    # left/right/below the box specifically (where a parked vehicle body
    # would sit relative to a sign/panel that's actually its side/rear).
    for (sx1, sy1, sx2, sy2) in [
        (max(0, x - mx), y, x, y + h),                       # strip to the left
        (x + w, y, min(frame_w, x + w + mx), y + h),         # strip to the right
        (x, y + h, x + w, min(frame_h, y + h + my)),         # strip below
    ]:
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        strip = seg_map[sy1:sy2, sx1:sx2]
        if strip.size == 0:
            continue
        if float(np.isin(strip, list(ED.VEHICLE_IDS)).mean()) > VEHICLE_RING_REJECT:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════
# 4. Cross-frame tracking
# ═════════════════════════════════════════════════════════════════════════

def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


APPEARANCE_SIM_MIN = 0.45   # min HSV-histogram correlation (0-1) required to accept a
                             # match that has little/no IoU and is relying mainly on the
                             # proximity fallback below — without this, any two candidates
                             # of the same broad group (e.g. two unrelated buildings, both
                             # bucketed as "wall") that happen to sit in roughly the same
                             # screen region get linked into one track just because of
                             # position, even though they're visibly different surfaces.


def _color_hist(hsv_frame, bbox):
    """Cheap appearance signature for a candidate box: normalized H/S
    histogram of its region. Used only to sanity-check that two
    same-group candidates in different frames plausibly show the same
    physical surface (not two different objects that merely occupy a
    similar screen position)."""
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, x); y = max(0, y)
    roi = hsv_frame[y:y + max(1, h), x:x + max(1, w)]
    if roi.size == 0:
        return None
    hist = cv2.calcHist([roi], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _hist_sim(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return max(0.0, float(sim))


def _match_score(track_bbox, cand_bbox, frame_w, gap=1, appearance_sim=None):
    """Combines IoU with a center-distance/area-ratio fallback, since at
    1 fps a billboard can shift or grow a lot between sampled frames —
    plain IoU alone would lose the track too easily. `gap` is how many
    sampled frames since this track was last seen; tolerance widens for
    bigger gaps (e.g. after a brief occlusion) since more apparent motion
    is expected.

    When there's real box overlap (iou > 0) that's strong evidence on its
    own. When there's little/no overlap and the match would rely on the
    proximity fallback alone, that fallback is only honored if the two
    boxes also look alike (appearance_sim) — otherwise same-group
    candidates that just happen to share screen position (e.g. two
    different buildings passing by) would incorrectly link into one
    track."""
    iou = _iou_xywh(track_bbox, cand_bbox)
    tx, ty, tw, th = track_bbox
    cx_, cy_, cw, ch = cand_bbox
    tcx, tcy = tx + tw / 2, ty + th / 2
    ccx, ccy = cx_ + cw / 2, cy_ + ch / 2
    dist_norm = math.hypot(tcx - ccx, tcy - ccy) / max(1, frame_w)
    area_ratio = (cw * ch) / max(1.0, (tw * th))
    dist_limit = CENTER_DIST_NORM * max(1, gap)
    proximity_ok = (dist_norm < dist_limit and AREA_RATIO_LO <= area_ratio <= AREA_RATIO_HI)

    if iou <= 0.02 and proximity_ok:
        # relying (almost) entirely on position — require appearance to agree
        if appearance_sim is None or appearance_sim < APPEARANCE_SIM_MIN:
            proximity_ok = False

    proximity_bonus = 0.25 if proximity_ok else 0.0
    return iou + proximity_bonus


VIDEO_FOLIAGE_OCC_ALLOW = 0.55   # trees/branches are transient and swaying — the per-frame
                                  # visual score already penalizes occlusion continuously
                                  # (s_occ, weighted 0.35 of the total); a hard binary cutoff
                                  # on top of that throws away real signage before dwell-
                                  # tracking across frames even gets a chance to see it
NON_FOLIAGE_OCC_REJECT  = 0.20    # vehicles/people/poles/signs actually blocking the view
                                  # still hard-reject at the same bar as the image pipeline


def _relax_foliage_occlusion(c, seg_map):
    """The shared hard-reject gate (engine_driver.hard_reject_reason) rejects
    anything with >20% occluder coverage, lumping tree branches in with
    vehicles/people/poles. That's the right call for a single still photo
    — but it's too strict for a driving video down a tree-lined street:
    foliage sways, and a branch crossing a sign in one sampled frame often
    doesn't in the next. Re-admit a candidate that was hard-rejected
    *only* for occlusion, provided the occlusion is mostly foliage (not a
    vehicle/person/pole genuinely blocking the view) and stays under a
    looser cap. The per-frame visual score still penalizes the occlusion
    continuously — this just stops it from being thrown out before
    tracking/dwell scoring ever sees it."""
    reason = c["hard_reject"]
    if reason is None:
        return True
    if not reason.startswith("heavily occluded"):
        return False
    x, y, w, h = c["bbox"]
    roi = seg_map[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    foliage = float(np.isin(roi, list(ED.TREE_IDS)).sum()) / roi.size
    non_foliage_ids = ED.OCCLUDER_IDS - ED.TREE_IDS
    non_foliage = float(np.isin(roi, list(non_foliage_ids)).sum()) / roi.size
    return non_foliage <= NON_FOLIAGE_OCC_REJECT and (foliage + non_foliage) <= VIDEO_FOLIAGE_OCC_ALLOW


def dedupe_nearby_candidates(scored, proximity_factor=1.0):
    """A single small sign/decal can get segmented into two or three
    disconnected blobs — split by a wire, a pole, or just a rough edge —
    each of which independently clears the size threshold and becomes
    its own candidate. Cross-frame tracking would normally sort this out
    (the fragments would merge into one track once IoU/proximity across
    frames links them), but a fragment that only shows up in a single
    sampled frame never gets that chance, and ends up surfaced as its
    own separate 'site' — a near-duplicate of the real one. Within a
    single frame, collapse same-group candidates whose centers are close
    together relative to their own size, keeping only the best-scoring
    fragment."""
    order = sorted(range(len(scored)), key=lambda i: scored[i]["score"], reverse=True)
    used = [False] * len(scored)
    kept = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        kept.append(scored[i])
        xi, yi, wi, hi = scored[i]["bbox"]
        cxi, cyi = xi + wi / 2, yi + hi / 2
        diag_i = math.hypot(wi, hi)
        gi = _surface_group(scored[i]["class_name"])
        for j in order:
            if used[j] or j == i:
                continue
            if _surface_group(scored[j]["class_name"]) != gi:
                continue
            xj, yj, wj, hj = scored[j]["bbox"]
            cxj, cyj = xj + wj / 2, yj + hj / 2
            diag_j = math.hypot(wj, hj)
            dist = math.hypot(cxi - cxj, cyi - cyj)
            if dist < proximity_factor * max(diag_i, diag_j):
                used[j] = True
    return kept


def build_tracks(per_frame_scored, total_frames):
    """per_frame_scored: list (len = total_frames) of lists of scored
    candidate dicts (engine_driver.score_candidate output) for that
    frame. Returns a list of track dicts."""
    tracks = []  # each: {group, last_bbox, last_frame, frame_w, last_hist, hits:[(frame_idx, cand)]}

    for frame_idx, candidates in enumerate(per_frame_scored):
        # Build every valid (track, candidate) pairing for this frame and
        # its match score, THEN assign highest-scoring pairs first, globally
        # — instead of looping over tracks in insertion order and letting
        # whichever track comes first in the list "claim" the best-fitting
        # candidate even when it actually fits a different track better.
        # That list-order greediness was letting one track steal a
        # candidate away from the track it truly belonged to whenever
        # several similar surfaces (e.g. multiple buildings) were in frame
        # at once.
        pairs = []  # (score, track_idx, cand_idx)
        for ti, track in enumerate(tracks):
            gap = frame_idx - track["last_frame"]
            if gap > MAX_FRAME_GAP:
                continue  # track has gone cold, don't extend it further
            for j, c in enumerate(candidates):
                if _surface_group(c["class_name"]) != track["group"]:
                    continue
                sim = _hist_sim(track.get("last_hist"), c.get("_hist"))
                sc = _match_score(track["last_bbox"], c["bbox"], track["frame_w"],
                                   gap=gap, appearance_sim=sim)
                if sc >= IOU_MATCH_MIN:
                    pairs.append((sc, ti, j))

        pairs.sort(key=lambda p: p[0], reverse=True)
        matched_tracks, matched_cands = set(), set()
        for sc, ti, j in pairs:
            if ti in matched_tracks or j in matched_cands:
                continue
            matched_tracks.add(ti)
            matched_cands.add(j)
            track = tracks[ti]
            c = candidates[j]
            track["hits"].append((frame_idx, c))
            track["last_bbox"] = c["bbox"]
            track["last_frame"] = frame_idx
            track["last_hist"] = c.get("_hist")

        # anything left over starts a brand-new track
        for j, c in enumerate(candidates):
            if j in matched_cands:
                continue
            tracks.append(dict(
                group=_surface_group(c["class_name"]),
                last_bbox=c["bbox"], last_frame=frame_idx, frame_w=c.get("_frame_w"),
                last_hist=c.get("_hist"),
                hits=[(frame_idx, c)],
            ))

    for t in tracks:
        t["dwell_score"] = round(len(t["hits"]) / total_frames, 3)
    return tracks


# ═════════════════════════════════════════════════════════════════════════
# 5. Final ranking (visual score blended with dwell)
# ═════════════════════════════════════════════════════════════════════════

def finalize_tracks(tracks):
    results = []
    for t in tracks:
        scores = [c["score"] for _, c in t["hits"]]
        k = min(3, len(scores))
        top_k = sorted(scores, reverse=True)[:k]
        base_score = sum(top_k) / len(top_k)

        final = base_score * (1 - DWELL_WEIGHT) + t["dwell_score"] * DWELL_WEIGHT
        final = round(min(ED.MAX_REALISTIC_SCORE, final), 4)

        n_frames_seen = len(t["hits"])
        # A surface glimpsed in only one sampled frame hasn't been
        # confirmed by tracking at all — it could be a one-off
        # segmentation flicker (e.g. briefly mislabeling part of a
        # fence/panel), OR it could be a perfectly real billboard that
        # the vehicle simply passed quickly relative to the 1fps sample
        # rate — at normal driving speed a roadside sign is very often
        # only going to land in one sampled second no matter how real it
        # is. Those two cases need different treatment, not the same
        # blanket discount.
        #
        # History: v1 capped the score at THR_VIABLE + 0.02 (0.54) — a
        # ceiling, not a penalty, so any flicker that already cleared
        # 0.52 stayed VIABLE at exactly 0.54, producing 2-3 identical-
        # looking "padded" sites. v2 replaced that with a flat ×0.65
        # discount on ALL single-frame tracks — but that requires a
        # near-perfect pre-discount score (~0.80+) just to survive, so
        # it started killing genuinely good single-glimpse billboards
        # too (0 sites shown even when the frame-by-frame audit clearly
        # had clean detections).
        #
        # v3: gate on the underlying visual quality instead of frame
        # count alone. A single glimpse with a strong base score (clear
        # sightline, low occlusion, decent size — i.e. it looks like a
        # real sign, not noise) only takes a mild confidence discount.
        # A single glimpse with a marginal/weak base score is much more
        # likely to be exactly the kind of flicker this check exists to
        # catch, and gets pushed firmly into AVOID instead.
        low_confidence = n_frames_seen < 2
        if low_confidence:
            if base_score < UNCONFIRMED_MIN_BASE_SCORE:
                final = round(min(final, ED.THR_VIABLE - 0.05), 4)
            else:
                final = round(final * LOW_CONFIDENCE_PENALTY, 4)

        if final >= ED.THR_PREMIUM:
            rec = "PREMIUM"
        elif final >= ED.THR_VIABLE:
            rec = "VIABLE"
        else:
            rec = "AVOID"

        best_frame_idx, best_cand = max(t["hits"], key=lambda h: h[1]["score"])
        first_ts = min(c.get("_ts", 0) for _, c in t["hits"])
        last_ts = max(c.get("_ts", 0) for _, c in t["hits"])

        results.append(dict(
            class_name=best_cand["class_name"], is_ad=best_cand["is_ad"], is_building=best_cand["is_building"],
            base_score=round(base_score, 4), dwell_score=t["dwell_score"],
            score=final, recommendation=rec, low_confidence=low_confidence,
            n_frames_seen=n_frames_seen, seconds_visible=round(last_ts - first_ts + 1, 1),
            best_frame_idx=best_frame_idx, best_candidate=best_cand,
        ))

    # Prefer tracks the tracker actually confirmed across multiple frames
    # over single-frame ones, even if a single-frame one happens to score
    # marginally higher on paper — it's the more trustworthy read of what
    # was actually out there.
    results.sort(key=lambda r: (not r["low_confidence"], r["score"]), reverse=True)
    return results


def build_video_bullets(r, total_frames):
    c = r["best_candidate"]
    bullets = [
        f"Detected as {c['class_name']}, {c['h_zone']} of frame, "
        f"{c['size_label'].lower()} surface (~{c['area_pct']:.1f}% of frame area at best view)",
        f"Visible in {r['n_frames_seen']} of {total_frames} sampled frames "
        f"(dwell score {r['dwell_score']:.2f}) — spanning ~{r['seconds_visible']:.0f}s of the clip",
        f"Best single-frame visual score: {r['base_score']:.2f} "
        f"({c['occ_label']} sightline, occlusion {c['s_occ']:.2f})",
    ]
    if c["is_ad"] and not c.get("auto_detected"):
        bullets.append("Existing ad structure — already has installed infrastructure and likely legal clearance")
    else:
        bullets.append(f"Blank/flat surface quality: {c['blank_score']*100:.0f}%")
    if r.get("low_confidence"):
        bullets.append("⚠ Detected in only a single sampled frame — not confirmed by tracking across "
                        "the clip, treat as lower confidence and verify in person")
    return bullets


# ═════════════════════════════════════════════════════════════════════════
# 6. Drawing (one annotated frame per recommended site — its clearest view)
# ═════════════════════════════════════════════════════════════════════════

def draw_site_on_frame(frame_rgb, cand, rec):
    out = frame_rgb.copy()
    x, y, w, h = [int(v) for v in cand["bbox"]]
    col = ED.REC_COL.get(rec, (255, 165, 0))
    overlay = out.copy().astype(np.float32)
    fill = np.full_like(overlay[y:y+h, x:x+w], col, dtype=np.float32)
    overlay[y:y+h, x:x+w] = overlay[y:y+h, x:x+w] * 0.84 + fill * 0.16
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
    label = f"{rec}  {cand['score']:.2f}"
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    ly = max(th + 10, y - 4)
    cv2.rectangle(out, (x, ly - th - 10), (x + tw + 12, ly), (10, 10, 10), -1)
    cv2.putText(out, label, (x + 6, ly - 6), font, fs, col, ft, cv2.LINE_AA)
    return out


def draw_all_candidates_on_frame(frame_rgb, scored_candidates):
    """Frame-by-frame audit view: draws every candidate that passed the
    hard-reject gate on this specific sampled frame (not just the winning
    tracks), each labeled with its class and per-frame score, colour-coded
    by what tier it would fall into on its own. This is the transparency
    view — it lets you see exactly what the model found (and didn't
    reject) on every single sampled second, including anything sitting
    right next to a vehicle, so you can sanity-check the pipeline instead
    of only seeing the final top-N summary."""
    out = frame_rgb.copy()
    for c in scored_candidates:
        x, y, w, h = [int(v) for v in c["bbox"]]
        score = c["score"]
        if score >= ED.THR_PREMIUM:
            rec = "PREMIUM"
        elif score >= ED.THR_VIABLE:
            rec = "VIABLE"
        else:
            rec = "AVOID"
        col = ED.REC_COL.get(rec, (255, 165, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = f"{c['class_name']} {score:.2f}"
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        ly = max(th + 6, y - 3)
        cv2.rectangle(out, (x, ly - th - 6), (x + tw + 8, ly), (10, 10, 10), -1)
        cv2.putText(out, label, (x + 4, ly - 4), font, fs, col, ft, cv2.LINE_AA)
    if not scored_candidates:
        cv2.putText(out, "no surfaces passed the gate on this frame",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════

def analyze_video(video_path, tmp_dir, output_dir, job_id,
                   max_seconds=MAX_SECONDS, target_fps=TARGET_FPS):
    """
    Samples the video at ~1 fps (capped at max_seconds), runs the driver
    engine's own detection/scoring per frame, tracks candidates across
    frames, blends a dwell (occurrence-frequency) score into the final
    ranking, and returns the top 3-5 sites — each with its own best-view
    annotated frame saved to output_dir.
    """
    frames = extract_frames(video_path, tmp_dir, target_fps, max_seconds)
    if not frames:
        raise ValueError("No frames could be extracted from this video.")
    total_frames = len(frames)
    frame_h, frame_w = frames[0]["bgr"].shape[:2]

    seg_maps = segment_batch([f["path"] for f in frames])

    per_frame_scored = []
    debug_frames = []
    for f, seg_map in zip(frames, seg_maps):
        sky_mask = np.isin(seg_map, list(ED.SKY_IDS)).astype(np.uint8)
        road_mask = np.isin(seg_map, list(ED.ROAD_IDS)).astype(np.uint8)
        candidates = ED.extract_candidates(seg_map, f["bgr"], f["hsv"], frame_h, frame_w, sky_mask, road_mask)
        passed = [c for c in candidates if _relax_foliage_occlusion(c, seg_map)]
        # Segmentation occasionally mislabels a vehicle's flat side panel
        # as a wall/fence surface, which the standard occlusion check
        # can't catch (there's no vehicle-labeled pixel *inside* the box
        # for it to see). Filter those out before they're even scored.
        passed = [c for c in passed
                  if not _vehicle_context_reject(c["bbox"], seg_map, road_mask, frame_w, frame_h)]
        scored = [ED.score_candidate(c, seg_map, f["bgr"], frame_h, frame_w) for c in passed]
        for c in scored:
            c["_frame_idx"] = f["idx"]
            c["_frame_w"] = frame_w
            c["_ts"] = f["ts"]
            c["_hist"] = _color_hist(f["hsv"], c["bbox"])
        scored = dedupe_nearby_candidates(scored)
        per_frame_scored.append(scored)

        debug_img = draw_all_candidates_on_frame(f["rgb"], scored)
        debug_path = os.path.join(output_dir, f"{job_id}_frame{f['idx']:03d}.jpg")
        cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        debug_frames.append(dict(
            idx=f["idx"], ts=f["ts"],
            filename=os.path.basename(debug_path),
            n_candidates=len(scored),
        ))

    tracks = build_tracks(per_frame_scored, total_frames)
    ranked = finalize_tracks(tracks)

    # Never pad results with AVOID-tier tracks just to have something to
    # show — a closed shop shutter or a bare wall scoring AVOID 0.5 is
    # not a real recommendation, and showing it anyway (even labeled as
    # a "fallback") is exactly the kind of forced-quota behavior this is
    # meant to avoid. Show every non-AVOID track found, up to TOP_MAX —
    # that may be 0, 1, 2, or up to 5 sites depending on what the clip
    # actually contains. If it's 0, the results page says so plainly
    # instead of showing anything.
    non_avoid = [r for r in ranked if r["recommendation"] != "AVOID"]
    top = non_avoid[:TOP_MAX]

    for i, r in enumerate(top, 1):
        r["sid"] = f"SITE-{i:03d}"
        r["bullets"] = build_video_bullets(r, total_frames)
        best_frame = frames[r["best_frame_idx"]]
        annotated = draw_site_on_frame(best_frame["rgb"], r["best_candidate"], r["recommendation"])
        thumb_path = os.path.join(output_dir, f"{job_id}_site{i}.jpg")
        cv2.imwrite(thumb_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        r["thumb_filename"] = os.path.basename(thumb_path)

    return dict(
        frame_w=frame_w, frame_h=frame_h,
        total_frames=total_frames, sampled_fps=target_fps,
        sites=top,
        debug_frames=debug_frames,
    )
