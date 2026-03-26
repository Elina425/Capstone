"""Joint angles from 2D keypoints (degrees)."""

from __future__ import annotations

import numpy as np


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b formed by segments (a,b) and (c,b). Returns nan if degenerate."""
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angles_from_coco17_xy(
    xy: np.ndarray,
) -> np.ndarray:
    """
    xy: (17, 2) in image or normalized space (angles are invariant to uniform scale in 2D
    when all three points scale together; we use raw 2D triangle geometry).
    Order matches config.COCO_NAMES.
    Returns fixed-length vector of biomechanical angles (degrees).
    """
    p = xy
    # Indices follow COCO 17
    LS, RS = 5, 6
    LE, RE = 7, 8
    LW, RW = 9, 10
    LH, RH = 11, 12
    LK, RK = 13, 14
    LA, RA = 15, 16

    feats = [
        _angle_deg(p[LH], p[LK], p[LA]),  # left knee
        _angle_deg(p[RH], p[RK], p[RA]),  # right knee
        _angle_deg(p[LS], p[LH], p[LK]),  # left hip (shoulder–hip–knee)
        _angle_deg(p[RS], p[RH], p[RK]),  # right hip
        _angle_deg(p[LS], p[LE], p[LW]),  # left elbow
        _angle_deg(p[RS], p[RE], p[RW]),  # right elbow
        _angle_deg(p[LH], p[LS], p[LE]),  # left shoulder (hip–shoulder–elbow)
        _angle_deg(p[RH], p[RS], p[RE]),  # right shoulder
    ]
    return np.asarray(feats, dtype=np.float32)


def angle_feature_dim() -> int:
    return 8
