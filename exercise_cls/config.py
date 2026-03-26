"""Constants: COCO-17 layout, window size, target FPS, paths."""

from __future__ import annotations

import os
from pathlib import Path

# --- Data (repo layout: squat vs overhead press as "barbell" class) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EXERCISE_DATA_ROOT", str(REPO_ROOT)))
VIDEO_DIRS = {
    "squat": DATA_ROOT / "videos_squat",
    "barbell": DATA_ROOT / "videos_ohp",
}
CLASS_NAMES = ("squat", "barbell")
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# --- Sequences ---
WINDOW_SIZE = 30
TARGET_FPS = 10.0

# --- Keypoints: COCO 17 (x, y) + confidence; shape (T, 17, 3) ---
NUM_JOINTS = 17
COCO_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Undirected edges for ST-GCN (COCO topology)
STGCN_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)

# Default artifacts
ARTIFACTS_DIR = DATA_ROOT / "artifacts"
SEQUENCES_DIR = ARTIFACTS_DIR / "sequences"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
