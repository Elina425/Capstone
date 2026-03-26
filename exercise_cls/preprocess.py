"""Normalize keypoints, impute missing joints, resample to target FPS."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def normalize_torso_scale(kps: np.ndarray) -> np.ndarray:
    """
    kps: (T, 17, 3) with x,y in pixels, conf in [0,1].
    Scale by torso length (mean shoulder–hip distance) and center on hip midpoint.
    """
    t = kps.shape[0]
    out = kps.copy().astype(np.float32)
    ls, rs, lh, rh = 5, 6, 11, 12
    for i in range(t):
        xy = out[i, :, :2]
        conf = out[i, :, 2]
        hip_mid = (xy[lh] + xy[rh]) / 2.0
        sh_mid = (xy[ls] + xy[rs]) / 2.0
        scale = (np.linalg.norm(xy[ls] - xy[lh]) + np.linalg.norm(xy[rs] - xy[rh])) / 2.0
        if not np.isfinite(scale) or scale < 1e-3:
            scale = 1.0
        out[i, :, 0] = (xy[:, 0] - hip_mid[0]) / scale
        out[i, :, 1] = (xy[:, 1] - hip_mid[1]) / scale
        out[i, :, 2] = conf
    return out


def impute_missing(kps: np.ndarray, conf_thresh: float = 0.2) -> np.ndarray:
    """
    Mark low-confidence joints as nan, then linear interpolate along time, then forward/back fill.
    kps: (T, 17, 3)
    """
    x = kps.copy().astype(np.float32)
    t, j, _ = x.shape
    mask_bad = x[:, :, 2] < conf_thresh
    x[:, :, 0] = np.where(mask_bad, np.nan, x[:, :, 0])
    x[:, :, 1] = np.where(mask_bad, np.nan, x[:, :, 1])
    for ji in range(j):
        for dim in range(2):
            s = x[:, ji, dim]
            if np.all(np.isnan(s)):
                x[:, ji, dim] = 0.0
                continue
            idx = np.arange(t, dtype=np.float32)
            good = ~np.isnan(s)
            if good.sum() == 0:
                x[:, ji, dim] = 0.0
                continue
            if good.sum() == 1:
                x[:, ji, dim] = np.nanmean(s)
                continue
            x[:, ji, dim] = np.interp(idx, idx[good], s[good])
    x[:, :, 2] = np.where(np.isnan(x[:, :, 2]), 0.0, x[:, :, 2])
    return x


def resample_sequence(kps: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample (T, 17, 3) uniformly along time to new length for dst_fps given same duration."""
    t = kps.shape[0]
    if t < 2 or src_fps <= 0 or dst_fps <= 0:
        return kps
    duration = (t - 1) / src_fps
    new_len = max(2, int(round(duration * dst_fps)) + 1)
    old_t = np.linspace(0.0, 1.0, t)
    new_t = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((new_len, kps.shape[1], kps.shape[2]), dtype=np.float32)
    for ji in range(kps.shape[1]):
        for dim in range(3):
            f = interp1d(old_t, kps[:, ji, dim], kind="linear", fill_value="extrapolate")
            out[:, ji, dim] = f(new_t).astype(np.float32)
    return out


def preprocess_pipeline(
    kps: np.ndarray,
    src_fps: float,
    target_fps: float,
    conf_thresh: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Returns processed keypoints and effective fps (target_fps)."""
    x = impute_missing(kps, conf_thresh=conf_thresh)
    x = normalize_torso_scale(x)
    x = resample_sequence(x, src_fps=src_fps, dst_fps=target_fps)
    return x, target_fps
