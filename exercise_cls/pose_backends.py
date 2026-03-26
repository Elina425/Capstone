"""Pose estimation backends: MediaPipe BlazePose, Ultralytics YOLO-pose, OpenPose-style (optional)."""

from __future__ import annotations

import abc
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from exercise_cls.config import NUM_JOINTS, REPO_ROOT

# MediaPipe Pose (33 landmarks) -> COCO 17
_MP_TO_COCO = {
    0: 0,  # nose
    2: 1,  # left eye
    5: 2,  # right eye
    7: 3,  # left ear
    8: 4,  # right ear
    11: 5,
    12: 6,  # shoulders
    13: 7,
    14: 8,  # elbows
    15: 9,
    16: 10,  # wrists
    23: 11,
    24: 12,  # hips
    25: 13,
    26: 14,  # knees
    27: 15,
    28: 16,  # ankles
}


@dataclass
class PoseResult:
    """keypoints (17, 3) as x, y, confidence in pixel units."""

    keypoints: np.ndarray  # (17, 3)


class PoseBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def infer(self, bgr: np.ndarray) -> PoseResult:
        ...


_MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_mediapipe_pose_model() -> Path:
    cache = Path(os.environ.get("MEDIAPIPE_POSE_MODEL", ""))
    if cache and cache.is_file():
        return cache
    dest = REPO_ROOT / "artifacts" / "pose_landmarker_lite.task"
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MEDIAPIPE_MODEL_URL, dest)
    return dest


class MediaPipeBlazePoseBackend(PoseBackend):
    """MediaPipe Tasks Pose Landmarker (BlazePose), mapped to COCO-17."""

    name = "mediapipe_blazepose"

    def __init__(self) -> None:
        from mediapipe.tasks.python.core import base_options as bo
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.vision.core import image as image_lib
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as rm

        model_path = str(_ensure_mediapipe_pose_model())
        opts = PoseLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=model_path),
            running_mode=rm.VisionTaskRunningMode.IMAGE,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = PoseLandmarker.create_from_options(opts)
        self._Image = image_lib.Image
        self._ImageFormat = image_lib.ImageFormat

    def infer(self, bgr: np.ndarray) -> PoseResult:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._Image(self._ImageFormat.SRGB, rgb)
        res = self._landmarker.detect(mp_image)
        kps = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        if not res.pose_landmarks:
            return PoseResult(kps)
        lm = res.pose_landmarks[0]
        for mp_i, coco_i in _MP_TO_COCO.items():
            p = lm[mp_i]
            kps[coco_i] = (p.x * w, p.y * h, getattr(p, "visibility", 1.0))
        return PoseResult(kps)

    def close(self) -> None:
        self._landmarker.close()


class YoloPoseBackend(PoseBackend):
    """Ultralytics YOLOv8/YOLO11 pose; outputs COCO keypoints."""

    name = "yolo_pose"

    def __init__(self, model_name: str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_name or os.environ.get("YOLO_POSE_MODEL", "yolo11n-pose.pt"))

    def infer(self, bgr: np.ndarray) -> PoseResult:
        r = self._model.predict(bgr, verbose=False)[0]
        kps = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
            return PoseResult(kps)
        # first person
        xy = r.keypoints.data[0].cpu().numpy()
        if xy.shape[0] < NUM_JOINTS:
            return PoseResult(kps)
        kps[:, :2] = xy[:NUM_JOINTS, :2]
        kps[:, 2] = xy[:NUM_JOINTS, 2] if xy.shape[1] > 2 else 1.0
        return PoseResult(kps)


def get_backend(name: str) -> PoseBackend:
    n = name.lower().strip()
    if n in ("mediapipe", "blazepose", "mediapipe_blazepose"):
        return MediaPipeBlazePoseBackend()
    if n in ("yolo", "yolov8", "yolo11", "yolo_pose", "ultralytics"):
        return YoloPoseBackend()
    raise ValueError(f"Unknown backend: {name} (use mediapipe_blazepose or yolo_pose)")


def openpose_available() -> bool:
    try:
        import openpose  # noqa: F401
    except Exception:
        return False
    return True


def make_openpose_backend() -> PoseBackend | None:
    """OpenPose Python bindings are environment-specific; not implemented by default."""
    return None
