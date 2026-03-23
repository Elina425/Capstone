# Exercise Quality Assessment - Capstone Project Plan

## Paper Reference
**Title:** 2202.14019 - Exercise Quality Assessment from Video Using Pose Estimation

This project aims to assess the quality of exercises (Squats & OHP) using pose estimation and deep learning.

---

## Project Overview

### Exercise Types (2)
- **OHP (Overhead Press)** - ~3000+ videos
- **Squat** - ~3000+ videos  

### Error Types
- **OHP**: Error Elbows, Error Knees
- **Squat**: Error Knees Forward, Error Knees Inward

### Current Dataset Status
✅ **Videos**: Available for both exercises  
✅ **Error Labels**: Temporal annotations (time ranges when errors occur)  
✅ **Train/Val/Test Splits**: Already defined in JSON files  
⚠️ **Keypoints**: Not extracted yet

---

## Task 1: Dataset Verification & Organization (End of Week)

### Objectives
1. ✅ Verify video-label correspondence
2. ✅ Validate temporal annotation format
3. Analyze label distribution (error prevalence)
4. Document dataset statistics

### Current Status
- **Videos**: Organized in `/videos_ohp/` and `/videos_squat/`
- **Labels**: JSON files with temporal error ranges
- **Splits**: Train/val/test keys defined

### Output Files
- `dataset_analysis.json` - Statistics on videos, labels, and splits
- `dataset_validation_report.md` - Quality assurance report

---

## Task 2: Pose Estimation Models Comparison (Week 2-3)

### Models to Compare

#### 1. **YOLOv11** (Real-time, SOTA)
- Pros: Fast, accurate, widely available
- Cons: Lower precision on edge cases
- Speed: ~30-60 fps on GPU
- Output: 17 keypoints (COCO format)

#### 2. **OpenPose** (Traditional, Multi-person)
- Pros: Multi-person detection, robust
- Cons: Slower, higher computational cost
- Speed: ~10-20 fps on GPU
- Output: 18 keypoints (custom format)

#### 3. **MediaPipe BlazePose** (Edge-optimized)
- Pros: Extremely fast, optimized for single person
- Cons: Less robust in crowded scenes
- Speed: ~100+ fps on GPU
- Output: 33 keypoints (Mediapipe format)

#### 4. **Transformer-Based Models** (With DoRA/LoRA Optimization)
- **ViTPose** or **HRFormer**
- Pros: State-of-the-art accuracy, attention mechanisms
- Cons: Slower, higher memory requirements
- Speed: ~10-30 fps (variable)
- Output: 17-18 keypoints

### Comparison Criteria
✅ **Accuracy** (Joint localization error)  
✅ **Speed** (FPS on real data)  
✅ **Robustness** (Occlusions, viewpoints)  
✅ **Memory Usage**  
✅ **Consistency** (Temporal stability)  

### Real Data Testing
- Use random sample of 50 OHP videos
- Use random sample of 50 Squat videos
- Test each model on same hardware
- Measure inference time & memory

### Output
- `model_comparison_results.csv`
- `model_comparison_visualizations/` (keypoint overlays)
- `model_selection_report.md` (recommendation)

---

## Task 3: Keypoint Preprocessing Pipeline (Week 3-4)

### Steps

#### 3.1 **Keypoint Extraction**
- Use selected best-performing model
- Extract 17-21 joints per frame
- Output: `.pkl` or `.npy` per video

#### 3.2 **Normalization**
```
Normalize to person bounding box:
- Center of mass normalization
- Scale by shoulder-to-hip distance
- Handles camera distance variation
```

#### 3.3 **Imputation**
```
Handle missing/occluded joints:
- Linear interpolation for gaps < 5 frames
- Kalman filter for temporal smoothing
- Delete frames with > 30% missing joints
```

#### 3.4 **Frame Rate Synchronization**
```
Standardize all videos to 30 fps:
- Resample sequences
- Interpolate frames if needed
- Downsample if faster
```

#### 3.5 **Output Format**
- Shape: `(num_frames, 17, 2)` or `(num_frames, 17, 3)` with confidence
- Save as `/keypoints/` directory
- Metadata: video_id, original_fps, num_frames, errors

---

## Directory Structure

```
Capstone/
├── videos_ohp/           # OHP videos (~3000)
├── videos_squat/         # Squat videos (~3000)
├── Labels_ohp/           # Error labels
├── ohp_eval/             # Train/val/test splits
├── squat_eval/           # Train/val/test splits
├── keypoints/            # Extracted keypoints (output)
│   ├── ohp_keypoints/
│   └── squat_keypoints/
├── models/               # Pose estimation models
│   ├── yolov11.py
│   ├── openpose.py
│   ├── mediapipe.py
│   └── transformer_pose.py
├── utils/                # Utilities
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   └── evaluation.py
├── scripts/
│   ├── extract_keypoints.py
│   ├── compare_models.py
│   ├── preprocess_keypoints.py
│   └── analyze_dataset.py
├── results/              # Comparison results
│   ├── model_comparison_results.csv
│   └── model_comparison_visualizations/
└── PROJECT_PLAN.md       # This file
```

---

## Implementation Timeline

| Week | Task | Deliverables |
|------|------|---------------|
| **This Week** | Dataset verification | `dataset_analysis.json`, validation report |
| **Week 2-3** | Model comparison | Model comparison CSV, visualizations, selection |
| **Week 3-4** | Preprocessing pipeline | Normalized keypoints, preprocessing scripts |
| **Week 4+** | Model training & evaluation | Quality classifier, performance metrics |

---

## Key Decisions Needed

1. **Transformer Optimization**: Use LoRA, DoRA, or full fine-tuning?
2. **Keypoint Format**: 17 (COCO) or 18/33 (extended)?
3. **Confidence Threshold**: Accept keypoints with confidence > 0.5?
4. **GPU/Hardware**: What's available for comparisons?

---

## Paper Summary Notes

The paper addresses:
- Exercise quality assessment using pose estimation
- Real-time feedback for fitness applications
- Temporal error detection from keypoint sequences
- Multi-task learning for exercise type + quality

---

## References & Resources

- YOLOv11: [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenPose: [CMU Perceptual Computing Lab](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- MediaPipe: [Google](https://github.com/google/mediapipe)
- ViTPose: [GitHub](https://github.com/Megvii-BaseDetection/ViTPose)
- DoRA Paper: [arxiv](https://arxiv.org/abs/2402.09353)
- Paper: [2202.14019](https://arxiv.org/pdf/2202.14019)

