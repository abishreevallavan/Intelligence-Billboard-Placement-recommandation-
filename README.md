# Sightline — Billboard Site Recommender (Unified App)

Single Flask app that merges the two previously-separate projects
(driver-POV and pedestrian-POV billboard surface detection) into one
page with a toggle to switch modes.

## What changed vs. the two original apps

- `engine.py` from each project was kept **as-is** and renamed:
  - `engine_driver.py`      ← from the driver-POV app
  - `engine_pedestrian.py`  ← from the pedestrian-POV app
  No logic inside either engine was touched.
- `app.py` is new: it has a single `/analyze` route that reads a
  `mode` form field (`"driver"` or `"pedestrian"`) and dispatches to
  the matching engine's `analyze_image()`.
- `templates/index.html` is new: adds a pill-style toggle ("Driver POV"
  / "Pedestrian POV") above the upload dropzone. The selected mode is
  written into a hidden `<input name="mode">` that gets submitted with
  the form.
- `templates/results.html` shows a small badge with which mode was
  used, and only shows the debug "why rejected" table for pedestrian
  mode (matching the original pedestrian app's behavior).
- `static/style.css` is the original stylesheet with the toggle's CSS
  appended — same fonts/colors/spacing as before, nothing else changed.

## Run it

```
cd merged_app
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 — pick "Driver POV" or "Pedestrian POV",
then upload a photo as before.

## New: Video mode (Driver POV only)

On the upload page, when "Driver POV" is selected, a **Photo / Video**
sub-toggle appears. Video mode:

- Accepts MP4/MOV/AVI/WEBM/MKV, up to ~20 seconds (longer clips are
  truncated to the first 20s).
- Samples the clip at **1 frame/sec** (so a 20s clip → up to 20 frames)
  via `engine_driver_video.extract_frames()`.
- Runs the *same, unmodified* `engine_driver.py` segmentation →
  candidate extraction → hard-reject gate → scoring pipeline on every
  sampled frame — no scoring logic was duplicated or changed.
- Links each frame's candidates into cross-frame "tracks" (a billboard
  seen at second 3 and second 4 is recognized as the same physical
  surface, even though its position/size shifts as the vehicle moves).
- Adds **dwell score** = fraction of sampled frames a track stayed
  visible/trackable for. This is blended into the final ranking score
  (85% best-frames visual score + 15% dwell) alongside the existing
  occlusion/position/surface/size/shape factors — a surface visible for
  most of the drive-by outranks one glimpsed for a single frame, even
  if that single frame briefly scored well.
- Returns the **top 3-5 sites**, each shown with its own clearest
  ("best-view") annotated frame, plus how many of the sampled frames it
  was seen in.

For speed, frames are segmented in small batches (`BATCH_SIZE=4`)
through the same already-loaded SegFormer model rather than one-by-one,
and only 1 frame/sec is analyzed regardless of the source video's
actual frame rate — a 20s clip is at most 20 segmentation passes total.
If you need it faster still, lowering `TARGET_FPS` in
`engine_driver_video.py` (e.g. to 0.5 = one frame every 2s) cuts
compute roughly proportionally, at the cost of a coarser dwell measure.

## Fixes after first real-world test

Two issues showed up when actually run on a driving clip:

1. **Tracking was breaking almost every frame** ("visible in 1 of 13
   frames" for a surface that was in view the whole clip). Cause: the
   tracker required an *exact* class-name match between frames, but
   SegFormer's per-frame label for the same physical wall flickers
   between `wall` / `building` / `fence` — pure segmentation noise, not
   a real change. Fix: candidates are now matched by broad surface
   *group* (ad-like / wall-like / fence-like — see `_SURFACE_GROUPS` in
   `engine_driver_video.py`) instead of exact class name, and the
   spatial matching tolerance now scales up with how many frames a
   track has been missing (handles brief occlusion better too).

2. **A bus/van got recommended as a billboard surface.** Cause: the
   segmentation model occasionally mislabels a vehicle's flat side
   panel as `wall`/`fence` instead of `vehicle` — since the box's own
   pixels are (wrongly) not vehicle-labeled, the normal occlusion check
   has nothing to flag. Fix: `_vehicle_context_reject()` adds two cheap
   independent tells and rejects the candidate if either fires: (a) the
   box itself sits mostly on road-classified pixels (a fixed surface is
   set back from the road; a vehicle body is on it), or (b) a ring just
   outside the box has meaningful vehicle-classified pixels (e.g. the
   same vehicle's wheels/mirrors/windows that weren't misclassified).

These are heuristics, not a guarantee — segmentation-label noise and
vehicle misclassification can't be fully eliminated without a dedicated
video object tracker (e.g. SORT/DeepSORT with optical flow) or a
larger/finer-tuned segmentation model. If you still see occasional
false positives on certain footage, the constants to tune first are
`VEHICLE_RING_REJECT` / `ROAD_IN_BOX_REJECT` (vehicle guard) and
`CENTER_DIST_NORM` / `MAX_FRAME_GAP` (tracking) at the top of
`engine_driver_video.py`.

## Notes

- Both engines currently load their own copy of the SegFormer model on
  first use of that mode (lazy-loaded, cached after first call per
  engine module). Since it's the same base model
  (`nvidia/segformer-b5-finetuned-ade-640-640`), the weights are only
  *downloaded* once — but each engine module keeps its own in-memory
  copy once loaded, so total GPU/CPU memory use is roughly double what
  either original app used if you exercise both modes in one run.
  If that becomes a problem, the two `_load_model()` functions can be
  unified to share one model instance — say if you'd like that change.
