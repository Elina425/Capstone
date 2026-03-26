# Exercise classification (from scratch)

This repository implements a pipeline aligned with [arXiv:2411.11548](https://arxiv.org/abs/2411.11548) (BiLSTM on temporal angle features) and extends it with an **ST-GCN** branch on normalized joint coordinates.

## Classes and data

- **Squat** videos: `videos_squat/`
- **Barbell** (overhead press) videos: `videos_ohp/` — used as the “bar” exercise class for binary classification vs squat.

To fetch the Kaggle dataset (optional; place or symlink folders as above):

```python
import kagglehub
path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")
print(path)
```

Set `EXERCISE_DATA_ROOT` if your videos live outside the repo root.

## Pipeline

1. **Pose estimation** — [MediaPipe](https://developers.google.com/mediapipe) BlazePose or [Ultralytics](https://docs.ultralytics.com/) YOLO pose (COCO-17 keypoints per frame).
2. **Preprocess** — torso-scale normalization, confidence-based imputation, resampling to a fixed **10 FPS** so sequences are time-aligned.
3. **Biomechanics** — joint angles (knee, hip, elbow, shoulder) in degrees from 2D keypoints.
4. **Models**
   - **BiLSTM** on **30-frame** sliding windows of angle features.
   - **ST-GCN** on the same windows using normalized `(x, y, confidence)` per joint.

## Commands

```bash
pip install -r requirements.txt

# Compare pose backends (speed + mean keypoint confidence proxy)
python scripts/benchmark_pose.py --max-frames 120

# Extract NPZ windows + manifest.json (uses videos_squat / videos_ohp)
python scripts/extract_sequences.py --backend yolo_pose --stride 15

# Train (after extraction)
python scripts/train.py --model bilstm --epochs 50
python scripts/train.py --model stgcn --epochs 50
```

Artifacts are written under `artifacts/sequences/` and `artifacts/checkpoints/`.

## Notes

- **OpenPose** is not bundled; the benchmark script documents **MediaPipe** vs **YOLO pose**, which are portable in Python.
- Angle features are **view-sensitive in 2D** but more stable than raw pixels when combined with torso normalization, as in the referenced paper’s angle + coordinate design.
