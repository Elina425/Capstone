"""Video → per-frame keypoints → preprocess → angle sequences → sliding windows (NPZ)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from exercise_cls.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_JOINTS,
    SEQUENCES_DIR,
    TARGET_FPS,
    VIDEO_DIRS,
    WINDOW_SIZE,
)
from exercise_cls.geometry import angle_feature_dim, angles_from_coco17_xy
from exercise_cls.pose_backends import PoseBackend, get_backend
from exercise_cls.preprocess import preprocess_pipeline


def extract_keypoints_video(
    video_path: Path,
    backend: PoseBackend,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Returns keypoints (T, 17, 3) in pixel space, and source FPS.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pr = backend.infer(frame)
        frames.append(pr.keypoints)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    if not frames:
        return np.zeros((0, NUM_JOINTS, 3), dtype=np.float32), fps
    return np.stack(frames, axis=0).astype(np.float32), float(fps)


def keypoints_to_angle_sequence(kps_norm: np.ndarray) -> np.ndarray:
    """kps_norm: (T, 17, 3) normalized xy + conf — use xy for angles."""
    t = kps_norm.shape[0]
    ang = np.zeros((t, angle_feature_dim()), dtype=np.float32)
    for i in range(t):
        ang[i] = angles_from_coco17_xy(kps_norm[i, :, :2])
    return np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)


def sliding_windows(feat: np.ndarray, window: int, stride: int) -> np.ndarray:
    """feat: (T, F). Returns (N, window, F)."""
    t = feat.shape[0]
    if t < window:
        return np.zeros((0, window, feat.shape[1]), dtype=np.float32)
    out = []
    for start in range(0, t - window + 1, stride):
        out.append(feat[start : start + window])
    return np.stack(out, axis=0) if out else np.zeros((0, window, feat.shape[1]), dtype=np.float32)


def process_video_to_windows(
    video_path: Path,
    backend: PoseBackend,
    window: int = WINDOW_SIZE,
    stride: int = 15,
    target_fps: float = TARGET_FPS,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (windows_angle, windows_stgcn) where stgcn uses normalized xyz per joint (x,y,conf).
    """
    raw, src_fps = extract_keypoints_video(video_path, backend, max_frames=max_frames)
    if raw.shape[0] == 0:
        z = np.zeros((0, window, angle_feature_dim()), dtype=np.float32)
        return z, np.zeros((0, window, NUM_JOINTS, 3), dtype=np.float32)
    proc, _ = preprocess_pipeline(raw, src_fps=src_fps, target_fps=target_fps)
    angles = keypoints_to_angle_sequence(proc)
    win_ang = sliding_windows(angles, window, stride)
    # ST-GCN tensor: (N, T, V, C) with C=3 (x, y, conf)
    tw = sliding_windows(
        proc.reshape(proc.shape[0], -1), window, stride
    )  # (N, W, 17*3)
    n = tw.shape[0]
    st = tw.reshape(n, window, NUM_JOINTS, 3)
    return win_ang, st


def build_manifest(
    output_dir: Path,
    backend_name: str,
    stride: int = 15,
    limit_per_class: int | None = None,
    max_frames_per_video: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(backend_name)
    index: list[dict] = []
    try:
        for cls in CLASS_NAMES:
            vdir = VIDEO_DIRS[cls]
            if not vdir.is_dir():
                continue
            label = CLASS_TO_IDX[cls]
            taken = 0
            bar = tqdm(
                total=limit_per_class if limit_per_class is not None else None,
                desc=f"extract {cls}",
            )
            try:
                for vp in sorted(vdir.glob("*.mp4")):
                    if limit_per_class is not None and taken >= limit_per_class:
                        break
                    try:
                        wa, ws = process_video_to_windows(
                            vp,
                            backend,
                            stride=stride,
                            max_frames=max_frames_per_video,
                        )
                    except (RuntimeError, OSError):
                        continue
                    if wa.shape[0] == 0:
                        continue
                    taken += 1
                    bar.update(1)
                    stem = vp.stem
                    np.savez_compressed(
                        output_dir / f"{stem}_{cls}.npz",
                        angle_windows=wa,
                        stgcn_windows=ws,
                        label=np.int64(label),
                        source=str(vp),
                    )
                    index.append(
                        {
                            "file": f"{stem}_{cls}.npz",
                            "label": int(label),
                            "class": cls,
                            "windows": int(wa.shape[0]),
                        }
                    )
            finally:
                bar.close()
    finally:
        if hasattr(backend, "close"):
            backend.close()

    man_path = output_dir / "manifest.json"
    man_path.write_text(json.dumps(index, indent=2))
    return man_path


def load_split(
    manifest_path: Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified split by label so small manifests still get val/test when possible."""
    import random

    data = json.loads(manifest_path.read_text())
    random.seed(seed)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for row in data:
        by_label[row["label"]].append(row["file"])
    train, val, test = [], [], []
    for lab in (0, 1):
        files = by_label[lab][:]
        random.shuffle(files)
        n = len(files)
        if n == 0:
            continue
        if n == 1:
            train += files
        elif n == 2:
            train.append(files[0])
            test.append(files[1])
        elif n == 3:
            train.append(files[0])
            val.append(files[1])
            test.append(files[2])
        else:
            nt = max(1, int(round(n * train_ratio)))
            nv = max(1, int(round(n * val_ratio)))
            if nt + nv >= n:
                nv = max(1, n - nt - 1)
            if nt + nv >= n:
                nt = max(1, n - nv - 1)
            train += files[:nt]
            val += files[nt : nt + nv]
            test += files[nt + nv :]
    return train, val, test
