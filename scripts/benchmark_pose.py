#!/usr/bin/env python3
"""Compare pose backends on speed (FPS) and proxy detection quality on sample videos."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from exercise_cls.config import VIDEO_DIRS
from exercise_cls.pose_backends import get_backend


def _first_readable_video() -> Path | None:
    for d in VIDEO_DIRS.values():
        if not d.is_dir():
            continue
        for vp in sorted(d.glob("*.mp4")):
            cap = cv2.VideoCapture(str(vp))
            if cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    return vp
    return None


def mean_visible_confidence(keypoints: np.ndarray) -> float:
    """keypoints (T,17,3) conf in last channel."""
    return float(np.mean(keypoints[:, :, 2]))


def benchmark_video(video_path: Path, backend_name: str, max_frames: int = 120) -> dict:
    backend = get_backend(backend_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    t0 = time.perf_counter()
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        pr = backend.infer(frame)
        frames.append(pr.keypoints)
        n += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    if hasattr(backend, "close"):
        backend.close()
    if not frames:
        return {"error": "no frames", "backend": backend_name}
    kps = np.stack(frames, axis=0)
    proc_fps = n / max(elapsed, 1e-6)
    quality = mean_visible_confidence(kps)
    return {
        "backend": backend_name,
        "video": str(video_path),
        "frames": n,
        "seconds": elapsed,
        "inference_fps": proc_fps,
        "source_video_fps": fps_src,
        "mean_keypoint_confidence": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark_pose.json"))
    args = parser.parse_args()
    video = args.video
    if video is None:
        video = _first_readable_video()
        if video is None:
            raise SystemExit("No readable video found; pass --video")

    results = []
    for name in ("mediapipe_blazepose", "yolo_pose"):
        try:
            results.append(benchmark_video(video, name, max_frames=args.max_frames))
        except Exception as e:
            results.append({"backend": name, "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print("Wrote", args.out)
    for r in results:
        print(r)

    ok = [r for r in results if "inference_fps" in r]
    if len(ok) >= 2:
        best_speed = max(ok, key=lambda x: x["inference_fps"])
        best_qual = max(ok, key=lambda x: x["mean_keypoint_confidence"])
        print(
            "\nFastest (inference FPS):",
            best_speed["backend"],
            f"({best_speed['inference_fps']:.1f} FPS)",
        )
        print(
            "Highest mean confidence (proxy):",
            best_qual["backend"],
            f"({best_qual['mean_keypoint_confidence']:.3f})",
        )
        print(
            "Choose based on your use case: real-time latency favors speed; "
            "downstream accuracy may favor higher confidence when both are similar.",
        )


if __name__ == "__main__":
    main()
