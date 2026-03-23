# Exercise Quality Assessment System

## Overview

This capstone project aims to develop an AI system for assessing the quality of fitness exercises (Overhead Press and Squats) using pose estimation and machine learning. The system can detect common form errors and provide real-time feedback.

**Reference Paper:** [2202.14019 - Exercise Quality Recognition from Video](https://arxiv.org/pdf/2202.14019)

---

## Project Structure

```
Capstone/
├── videos_ohp/                      # OHP exercise videos (~3000)
├── videos_squat/                    # Squat exercise videos (~3000)
├── Labels_ohp/                      # Error annotations for OHP
├── errors_squat/                    # Error annotations for Squats
├── ohp_eval/ & squat_eval/         # Train/val/test split definitions
│
├── scripts/
│   ├── analyze_dataset.py           # Dataset verification & analysis
│   ├── compare_models.py            # Model comparison framework
│   ├── extract_keypoints.py         # Keypoint extraction (TODO)
│   ├── preprocess_keypoints.py      # Preprocessing pipeline (TODO)
│   └── train_classifier.py          # Exercise quality classifier (TODO)
│
├── pose_estimators.py               # Pose estimation model wrappers
├── keypoint_preprocessor.py         # Preprocessing utilities
├── requirements.txt                 # Python dependencies
├── PROJECT_PLAN.md                  # Detailed project plan
└── README.md                        # This file
```

---

## Quick Start

### Setup

1. **Clone/Navigate to project:**
   ```bash
   cd /Users/emelkonyan/PycharmProjects/Capstone
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) For GPU support:**
   ```bash
   # For CUDA 11.8 (adjust CUDA version as needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Run Dataset Analysis

Verify your dataset and understand label distribution:

```bash
python analyze_dataset.py
```

Output: `dataset_analysis.json` with comprehensive statistics

### Compare Pose Estimation Models

Test different models on your real data:

```bash
python compare_models.py \
    --capstone-root /Users/emelkonyan/PycharmProjects/Capstone \
    --num-samples 5 \
    --device cuda \
    --output-dir results/model_comparison
```

This will:
- ✅ Test YOLOv11M on 5 OHP videos + 5 Squat videos
- ✅ Test MediaPipe BlazePose on same videos
- ✅ Test ViTPose (Transformer) with DoRA optimization
- ✅ Generate comparison CSV and recommendations

---

## Tasks & Timeline

### Task 1: Dataset Verification ✅ (This Week)

**Status:** In Progress

- [x] Verify video-label correspondence
- [x] Validate error labels format
- [ ] Generate dataset statistics report
- [ ] Document label distribution

**Scripts:** `analyze_dataset.py`

**Expected Output:**
- `dataset_analysis.json` - Complete dataset statistics
- Validation report with data quality metrics

---

### Task 2: Model Comparison (Week 2-3)

**Status:** Implementation in Progress

**Models to Compare:**
1. **YOLOv11** - Real-time, SOTA general detection
2. **MediaPipe BlazePose** - Fast, edge-optimized
3. **Transformer-based (ViTPose)** - State-of-the-art accuracy
   - With **DoRA optimization** for efficient fine-tuning
   - With **LoRA option** for parameter-efficient adaptation

**Comparison Criteria:**
- Speed (FPS on GPU)
- Accuracy (keypoint localization error)
- Robustness (temporal consistency, occlusions)
- Memory usage
- Real-time feasibility

**Scripts:** `compare_models.py`

**Expected Outputs:**
- `model_comparison_results.csv` - Detailed metrics
- `recommendations.txt` - Model selection guidance
- Visualization images (keypoint overlays)

---

### Task 3: Keypoint Preprocessing (Week 3-4)

**Status:** Framework Ready

**Steps:**
1. **Normalization** - Handle camera distance variation
   - Center of mass normalization
   - Scale by shoulder-to-hip distance
   
2. **Imputation** - Handle missing/occluded joints
   - Linear interpolation for gaps
   - Kalman filtering for temporal smoothing
   - Remove frames with >30% missing joints
   
3. **Frame Rate Synchronization** - Standardize to 30 fps
   - Resample sequences
   - Interpolate/downsample as needed
   
4. **Output** - Standardized format
   - Shape: `(num_frames, 17, 2)` or `(num_frames, 17, 3)`
   - Saved as pickle/numpy files

**Scripts:** `keypoint_preprocessor.py` (Framework ready)

---

## Module Documentation

### `pose_estimators.py`

Unified interface for different pose estimation models.

**Classes:**
- `PoseEstimator` - Abstract base class
- `YOLOv11Pose` - YOLO model wrapper
- `MediaPipeBlazePose` - MediaPipe integration
- `ViTPose` - Transformer-based with DoRA support
- `ModelComparison` - Compare multiple models

**Usage:**
```python
from pose_estimators import YOLOv11Pose

model = YOLOv11Pose(model_size="m", device="cuda")
result = model.process_video("videos_ohp/sample.mp4", max_frames=300)
print(f"FPS: {result['avg_inference_fps']:.2f}")
```

### `keypoint_preprocessor.py`

Preprocess extracted keypoints for model training.

**Class:** `KeypointPreprocessor`

**Methods:**
- `normalize_keypoints()` - Normalize to handle camera distance
- `handle_missing_keypoints()` - Impute missing joints
- `synchronize_frame_rate()` - Resample to target fps
- `full_pipeline()` - Complete preprocessing

**Usage:**
```python
from keypoint_preprocessor import KeypointPreprocessor

preprocessor = KeypointPreprocessor(target_fps=30)
result = preprocessor.full_pipeline(
    keypoints=raw_keypoints,
    original_fps=30.0,
    normalize=True,
    impute=True,
    smooth=True,
    resample=True
)
```

### `analyze_dataset.py`

Dataset verification and analysis.

**Class:** `DatasetAnalyzer`

**Methods:**
- `analyze_exercise()` - Analyze single exercise type
- `run_full_analysis()` - Complete dataset analysis
- `save_analysis()` - Save results to JSON
- `print_summary()` - Print human-readable summary

**Usage:**
```bash
python analyze_dataset.py
```

### `compare_models.py`

Model comparison framework for real data.

**Class:** `ExerciseModelComparison`

**Methods:**
- `get_sample_videos()` - Get random video samples
- `compare_on_exercise()` - Compare models on exercise type
- `run_full_comparison()` - Full comparison on all exercises
- `save_results()` - Save results to CSV
- `generate_recommendations()` - Generate model recommendations

**Usage:**
```bash
python compare_models.py --num-samples 10 --device cuda
```

---

## Dataset Overview

### Videos
- **OHP:** ~3000 videos (various quality levels)
- **Squat:** ~3000 videos (various quality levels)

### Error Annotations
**OHP Errors:**
- `error_elbows.json` - Elbow form errors with temporal ranges
- `error_knees.json` - Knee form errors with temporal ranges

**Squat Errors:**
- `error_knees_forward.json` - Knees extending too far forward
- `error_knees_inward.json` - Knees caving inward

**Format:** `{"video_id": [[start_time, end_time], ...], ...}`

### Train/Val/Test Splits
Located in `ohp_eval/` and `squat_eval/`:
- `train_keys.json`
- `val_keys.json`
- `test_keys.json`

---

## Key Implementation Details

### 17 COCO Keypoints
```
0: Nose
1-2: Eyes (left, right)
3-4: Ears (left, right)
5-6: Shoulders (left, right)
7-8: Elbows (left, right)
9-10: Wrists (left, right)
11-12: Hips (left, right)
13-14: Knees (left, right)
15-16: Ankles (left, right)
```

### Normalization Strategies
1. **Center-Scale:** Normalize to shoulder-to-hip distance
2. **Bounding Box:** Normalize to person's bounding box
3. **Relative:** Normalize relative to nose position

### Missing Data Handling
1. **Linear Interpolation:** For gaps < 5 frames
2. **Kalman Filter:** For temporal smoothing
3. **Forward Fill:** Simple forward/backward fill

---

## Installation Notes

### For YOLOv11:
```bash
pip install ultralytics
```

### For MediaPipe (optional):
```bash
pip install mediapipe
```

### For ViTPose/Transformer Models (optional):
```bash
pip install timm
# Or for full MMPose framework:
pip install mmpose
```

### For GPU Support:
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Paper Summary

**Reference:** [Exercise Quality Assessment from Video using Pose Estimation and Deep Learning](https://arxiv.org/pdf/2202.14019)

**Key Points:**
- Uses pose estimation to extract body keypoints from videos
- Temporal CNN/RNN models to classify exercise quality
- Multi-task learning: exercise type + quality classification
- Real-time feedback system for fitness applications
- Handles multiple exercise types and form errors

**Our Approach:**
- Extract 17 COCO keypoints using best-performing model
- Normalize and preprocess keypoints
- Train classifier on labeled data
- Deploy for real-time assessment

---

## Results & Metrics

After running the pipeline, check:
- `dataset_analysis.json` - Dataset statistics
- `results/model_comparison/` - Model performance metrics
- `results/model_comparison/recommendations.txt` - Model selection

---

## Next Steps

1. ✅ Complete Task 1: Dataset verification
2. 🔄 Task 2: Run model comparison on real data
3. 🔄 Task 3: Extract and preprocess keypoints
4. ⏳ Task 4: Train quality classification model
5. ⏳ Task 5: Evaluate on test set & create demo

---

## Troubleshooting

### GPU Memory Issues
```python
# Reduce batch size or use CPU
python compare_models.py --device cpu
```

### Missing Keypoints from Model
Check confidence threshold and increase max_frames for more samples.

### Video Processing Slow
Reduce number of frames in testing:
```python
result = model.process_video(video_path, max_frames=100)
```

---

## Contributors & References

- **Project:** Exercise Quality Assessment Capstone
- **Advisor:** [Your Professor]
- **Paper Reference:** [2202.14019](https://arxiv.org/pdf/2202.14019)
- **Pose Estimation Models:**
  - YOLOv11: [Ultralytics](https://github.com/ultralytics/ultralytics)
  - MediaPipe: [Google](https://github.com/google/mediapipe)
  - ViTPose: [Megvii](https://github.com/Megvii-BaseDetection/ViTPose)
  - DoRA Paper: [arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)

---

## License

[Your License Here]

---

## Questions?

For issues or questions, refer to `PROJECT_PLAN.md` for detailed task breakdown.
