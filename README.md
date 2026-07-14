
# Intelligence-Billboard-Placement-recommandation-

An AI-powered Surface Analysis for billboard placement recommendation system that analyzes street-view images and identifies optimal advertising locations using semantic segmentation and computer vision.

The system evaluates candidate billboard surfaces based on visibility, surface quality, occlusion, size, position, and temporal consistency, helping advertisers identify high-impact billboard locations

---

## Features

- Driver Perspective Analysis
- Pedestrian Perspective Analysis
- Image Upload
- Video Upload (Driver Mode)
- Semantic Surface Segmentation
- Billboard Candidate Detection
- Occlusion Analysis
- Surface Quality Evaluation
- Visibility Scoring
- Temporal Tracking Across Frames
- Annotated Output Generation

---

## Technologies Used

- Python
- Flask
- PyTorch
- HuggingFace Transformers
- OpenCV
- SegFormer (ADE20K)
- HTML
- CSS
- JavaScript

---

## Project Structure

```
merged_app_patched/
│
├── app.py
├── engine_driver.py
├── engine_driver_video.py
├── engine_pedestrian.py
├── requirements.txt
├── README.md
│
├── static/
├── templates/
├── outputs/
└── frames_tmp/
```

---

## Installation

```bash
git clone https://github.com/USERNAME/intelligent-billboard-placement-recommendation.git

cd intelligent-billboard-placement-recommendation

pip install -r requirements.txt

python app.py
```

---

## Usage

1. Start the Flask server.
3. Open `http://127.0.0.1:5000`.
4. Choose Driver or Pedestrian mode.
5. Upload an image or video.
6. View recommended billboard locations with annotated results.

---

## Scoring Parameters

The recommendation score considers:

- Surface suitability
- Visible area
- Occlusion percentage
- Position in the frame
- Billboard dimensions
- Shape quality
- Temporal consistency (video mode)

---

