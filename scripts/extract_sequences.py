#!/usr/bin/env python3
"""Download Kaggle dataset (optional) and extract pose sequences to NPZ + manifest."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from exercise_cls.config import SEQUENCES_DIR
from exercise_cls.extract import build_manifest


def _maybe_download_kaggle() -> Path | None:
    try:
        import kagglehub
    except ImportError:
        return None
    path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")
    print("Path to dataset files:", path)
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="yolo_pose",
        help="mediapipe_blazepose or yolo_pose",
    )
    parser.add_argument("--stride", type=int, default=15, help="sliding window stride in frames")
    parser.add_argument("--out", type=Path, default=SEQUENCES_DIR)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--download-kaggle", action="store_true", help="download Kaggle dataset via kagglehub")
    args = parser.parse_args()

    if args.download_kaggle:
        _maybe_download_kaggle()

    man = build_manifest(
        args.out,
        backend_name=args.backend,
        stride=args.stride,
        limit_per_class=args.limit_per_class,
        max_frames_per_video=args.max_frames_per_video,
    )
    print("Wrote manifest:", man)


if __name__ == "__main__":
    main()
