# Quick Setup Checklist

This is a step-by-step checklist to get your project running. Complete these items in order.

## Environment Setup

- [ ] **Create virtual environment**
  ```bash
  cd /Users/emelkonyan/PycharmProjects/Capstone
  python -m venv venv
  source venv/bin/activate
  ```

- [ ] **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Verify PyTorch/CUDA (optional but recommended)**
  ```bash
  python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
  ```

## Task 1: Dataset Verification (This Week)

- [ ] **Run dataset analysis**
  ```bash
  python analyze_dataset.py
  ```
  - Creates `dataset_analysis.json`
  - Prints summary to console
  - Validates data integrity

- [ ] **Review analysis results**
  - Check number of videos per exercise
  - Verify error label distribution
  - Check for train/val/test overlaps
  - Review data quality issues

- [ ] **Document findings**
  - Note any data quality issues
  - Record dataset statistics
  - Identify missing videos (if any)

## Task 2: Model Comparison (Week 2-3)

### Prerequisites
- [ ] **Ensure you have good GPU** (recommended: RTX 3060 or better)
  - YOLOv11: ~4-6 GB VRAM
  - MediaPipe: ~2-3 GB VRAM
  - ViTPose: ~6-8 GB VRAM

### Installation
- [ ] **Install model-specific dependencies** (if not already installed)
  ```bash
  pip install ultralytics  # For YOLOv11
  # pip install mediapipe  # For MediaPipe (optional)
  ```

### Run Comparison
- [ ] **Start model comparison** (START WITH SMALL SAMPLE!)
  ```bash
  python compare_models.py \
    --num-samples 2 \
    --device cuda \
    --output-dir results/model_comparison
  ```

- [ ] **Review results**
  - Check `results/model_comparison/ohp_model_comparison.csv`
  - Check `results/model_comparison/squat_model_comparison.csv`
  - Read `results/model_comparison/recommendations.txt`

- [ ] **Expand testing** (if first run successful)
  ```bash
  python compare_models.py \
    --num-samples 50 \
    --device cuda \
    --output-dir results/model_comparison_full
  ```

## Task 3: Keypoint Preprocessing (Week 3-4)

- [ ] **Framework ready** - `keypoint_preprocessor.py` is implemented
- [ ] **Test preprocessing on sample** (create `test_preprocessing.py`)
- [ ] **Extract keypoints** using best model (to be created)
- [ ] **Apply full preprocessing pipeline**
- [ ] **Verify output format**

## Optional: Model Fine-tuning

- [ ] **ViTPose with DoRA** (for improved accuracy)
  - Research DoRA implementation
  - Adapt ViTPose model
  - Compare vs standard fine-tuning

- [ ] **Custom optimizer** (for unique exercise patterns)
  - Consider exercise-specific pose patterns
  - Implement domain adaptation

## File Checklist

Created files:
- [x] `README.md` - Main documentation
- [x] `PROJECT_PLAN.md` - Detailed planning
- [x] `requirements.txt` - Dependencies
- [x] `analyze_dataset.py` - Dataset analysis
- [x] `pose_estimators.py` - Model wrappers
- [x] `keypoint_preprocessor.py` - Preprocessing
- [x] `compare_models.py` - Model comparison
- [x] `SETUP_CHECKLIST.md` - This file

TODO files (to be created):
- [ ] `extract_keypoints.py` - Batch keypoint extraction
- [ ] `preprocess_keypoints_batch.py` - Batch preprocessing
- [ ] `train_classifier.py` - Quality classifier training
- [ ] `evaluate_model.py` - Model evaluation
- [ ] `demo.py` - Real-time demo

## Common Issues & Solutions

### Issue: CUDA out of memory
**Solution:**
```bash
python compare_models.py --device cpu --num-samples 2
```

### Issue: No videos found
**Solution:** Check paths are correct
```python
from pathlib import Path
Path("/Users/emelkonyan/PycharmProjects/Capstone/videos_ohp").exists()  # Should be True
```

### Issue: Model download fails
**Solution:** Manually download or check internet connection
```bash
# For YOLOv11, models auto-download on first run
# Ensure ~2GB free space
```

### Issue: ImportError for mediapipe
**Solution:** It's optional, can proceed without it
```bash
pip install mediapipe  # If needed
```

## Progress Tracking

### Week 1
- [ ] Day 1-2: Environment setup & dataset analysis
- [ ] Day 3-4: Model comparison (small sample)
- [ ] Day 5: Review results & expand testing

### Week 2-3
- [ ] Expand model comparison to full dataset
- [ ] Document model selection rationale
- [ ] Plan preprocessing pipeline

### Week 3-4
- [ ] Implement keypoint extraction
- [ ] Run preprocessing on full dataset
- [ ] Validate output quality

## Next Meeting Agenda

Before your next meeting/presentation, have ready:
1. Dataset analysis results (statistics, quality report)
2. Model comparison results (CSV with metrics)
3. Model selection recommendation
4. Preliminary keypoint extraction (sample output)
5. Any blockers or issues

## Contact & Resources

| Item | Link/Contact |
|------|--------------|
| Paper | https://arxiv.org/pdf/2202.14019 |
| YOLO | https://github.com/ultralytics/ultralytics |
| MediaPipe | https://github.com/google/mediapipe |
| ViTPose | https://github.com/Megvii-BaseDetection/ViTPose |
| DoRA Paper | https://arxiv.org/abs/2402.09353 |

---

**Last Updated:** March 20, 2026
**Status:** Setup Complete, Ready for Task 1
