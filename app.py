"""
app.py — Sightline: Billboard Site Recommender (unified web app)

Single application that supports two analysis modes, selected via a
toggle on the upload page:

  - "driver"     -> engine_driver.py     (driver's-eye-view street photos)
  - "pedestrian" -> engine_pedestrian.py (pedestrian's-eye-view street photos)

Each mode is backed by its own engine module, since the two pipelines
were tuned independently (different upscaling rules, window/balcony
detectors, etc.). The mode chosen on the upload form is passed through
to /analyze as a form field and determines which engine.analyze_image()
gets called.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in your browser.

NOTE: the first analysis will download the SegFormer model weights
(~350MB) from Hugging Face, so an internet connection is required the
first time the app runs. After that it's cached locally. Both engines
use the same base model name, so the weights are only downloaded once.
"""

import os
import shutil
import uuid
import traceback

from flask import Flask, render_template, request, send_from_directory

import engine_driver
import engine_pedestrian
import engine_driver_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FRAMES_DIR = os.path.join(BASE_DIR, "frames_tmp")
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "webm", "mkv"}

ENGINES = {
    "driver": engine_driver,
    "pedestrian": engine_pedestrian,
}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB (video needs headroom)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode", "driver")
    if mode not in ENGINES:
        mode = "driver"

    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose a file to upload.", mode=mode)

    is_video = allowed_video(file.filename)

    if is_video and mode != "driver":
        return render_template(
            "index.html",
            error="Video analysis is currently only available for Driver POV.",
            mode=mode,
        )

    if is_video:
        return analyze_video_upload(file, mode)

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload a JPG, PNG, WEBP image, or an MP4/MOV video (Driver POV only).",
            mode=mode,
        )

    engine = ENGINES[mode]
    job_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    file.save(upload_path)

    annotated_path = os.path.join(OUTPUT_DIR, f"{job_id}_annotated.jpg")

    try:
        result = engine.analyze_image(upload_path, annotated_path)
    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", error=f"Analysis failed: {e}", mode=mode)

    return render_template(
        "results.html",
        job_id=job_id,
        original_ext=ext,
        mode=mode,
        sites=result["sites"],
        frame_w=result["frame_w"],
        frame_h=result["frame_h"],
        num_rejected=len(result["rejected"]),
        num_avoid=len(result["avoid"]),
        rejected=result["rejected"],
        debug=(mode == "pedestrian"),
    )


def analyze_video_upload(file, mode):
    ext = file.filename.rsplit(".", 1)[1].lower()
    job_id = uuid.uuid4().hex[:12]
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    file.save(upload_path)

    job_frames_dir = os.path.join(FRAMES_DIR, job_id)

    try:
        result = engine_driver_video.analyze_video(
            upload_path, job_frames_dir, OUTPUT_DIR, job_id,
        )
    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", error=f"Video analysis failed: {e}", mode=mode)
    finally:
        shutil.rmtree(job_frames_dir, ignore_errors=True)

    return render_template(
        "results_video.html",
        job_id=job_id,
        mode=mode,
        sites=result["sites"],
        frame_w=result["frame_w"],
        frame_h=result["frame_h"],
        total_frames=result["total_frames"],
        sampled_fps=result["sampled_fps"],
        debug_frames=result["debug_frames"],
    )


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    # use_reloader=False is important: the reloader's file-watcher can
    # mistake PyTorch's own internal imports (e.g. torch/cuda/__init__.py)
    # for source-code changes and restart the server mid-analysis, which
    # kills the in-flight request (ERR_CONNECTION_RESET in the browser).
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
