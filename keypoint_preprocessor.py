"""
Keypoint Preprocessing Pipeline
Handles normalization, imputation, and frame rate synchronization
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeypointPreprocessor:
    """Preprocess extracted keypoints for model training"""
    
    # COCO17 keypoint indices
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, target_fps: int = 30, confidence_threshold: float = 0.3):
        """
        Initialize preprocessor
        
        Args:
            target_fps: Target frame rate for all videos
            confidence_threshold: Minimum confidence for valid keypoints
        """
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold
    
    def normalize_keypoints(
        self, 
        keypoints: np.ndarray,
        method: str = "center_scale"
    ) -> np.ndarray:
        """
        Normalize keypoints to handle camera distance variation
        
        Args:
            keypoints: Array of shape (num_frames, num_keypoints, 2 or 3)
            method: Normalization method ('center_scale', 'bounding_box', 'relative')
        
        Returns:
            Normalized keypoints
        """
        logger.info(f"Normalizing keypoints using {method}...")
        normalized = keypoints.copy()
        
        if method == "center_scale":
            # Normalize relative to shoulder-to-hip distance
            for frame_idx in range(keypoints.shape[0]):
                frame_kpts = keypoints[frame_idx, :, :2]  # (num_kpts, 2)
                
                # Get shoulder and hip positions
                # COCO indices: 5,6 = shoulders, 11,12 = hips
                if keypoints.shape[0] > 0:
                    left_shoulder = frame_kpts[5]
                    right_shoulder = frame_kpts[6]
                    left_hip = frame_kpts[11]
                    right_hip = frame_kpts[12]
                    
                    # Calculate center and scale
                    center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
                    
                    # Distance between shoulders and hips
                    shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
                    hip_dist = np.linalg.norm(right_hip - left_hip)
                    avg_dist = (shoulder_dist + hip_dist) / 2 + 1e-6
                    
                    # Normalize
                    normalized[frame_idx, :, :2] = (frame_kpts - center) / avg_dist
        
        elif method == "bounding_box":
            # Normalize to bounding box size
            for frame_idx in range(keypoints.shape[0]):
                frame_kpts = keypoints[frame_idx, :, :2]
                
                # Get bounding box
                valid_kpts = frame_kpts[~np.isnan(frame_kpts).any(axis=1)]
                if len(valid_kpts) > 0:
                    x_min, y_min = valid_kpts.min(axis=0)
                    x_max, y_max = valid_kpts.max(axis=0)
                    
                    width = x_max - x_min + 1e-6
                    height = y_max - y_min + 1e-6
                    
                    # Normalize
                    normalized[frame_idx, :, 0] = (frame_kpts[:, 0] - x_min) / width
                    normalized[frame_idx, :, 1] = (frame_kpts[:, 1] - y_min) / height
        
        elif method == "relative":
            # Normalize relative to nose position
            for frame_idx in range(keypoints.shape[0]):
                nose = keypoints[frame_idx, 0, :2]
                normalized[frame_idx, :, :2] = keypoints[frame_idx, :, :2] - nose
        
        return normalized
    
    def handle_missing_keypoints(
        self,
        keypoints: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        method: str = "linear_interp"
    ) -> np.ndarray:
        """
        Impute missing or low-confidence keypoints
        
        Args:
            keypoints: Array of shape (num_frames, num_keypoints, 2 or 3)
            confidence: Confidence scores if available
            method: Imputation method ('linear_interp', 'kalman', 'forward_fill')
        
        Returns:
            Imputed keypoints
        """
        logger.info(f"Handling missing keypoints using {method}...")
        imputed = keypoints.copy()
        
        # Mark invalid keypoints
        if confidence is not None:
            # Mark as NaN if confidence < threshold
            invalid_mask = confidence < self.confidence_threshold
            for frame_idx, kpt_idx in zip(*np.where(invalid_mask)):
                imputed[frame_idx, kpt_idx, :2] = np.nan
        
        if method == "linear_interp":
            # Linear interpolation for each keypoint
            for kpt_idx in range(keypoints.shape[1]):
                for coord in range(2):  # x, y
                    kpt_series = imputed[:, kpt_idx, coord]
                    valid_idx = ~np.isnan(kpt_series)
                    
                    if np.sum(valid_idx) > 1:
                        # Interpolate
                        f = interp1d(
                            np.where(valid_idx)[0],
                            kpt_series[valid_idx],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        
                        all_idx = np.arange(len(kpt_series))
                        imputed[:, kpt_idx, coord] = f(all_idx)
        
        elif method == "kalman":
            # Simple Kalman filter for temporal smoothing
            for kpt_idx in range(keypoints.shape[1]):
                for coord in range(2):
                    kpt_series = imputed[:, kpt_idx, coord]
                    
                    # Skip if mostly NaN
                    valid_count = np.sum(~np.isnan(kpt_series))
                    if valid_count < 3:
                        continue
                    
                    # Kalman filter implementation
                    filtered = self._kalman_filter_1d(kpt_series)
                    imputed[:, kpt_idx, coord] = filtered
        
        elif method == "forward_fill":
            # Forward and backward fill
            for kpt_idx in range(keypoints.shape[1]):
                for coord in range(2):
                    kpt_series = imputed[:, kpt_idx, coord]
                    # Forward fill
                    mask = np.isnan(kpt_series)
                    idx = np.where(~mask, np.arange(len(mask)), 0)
                    imputed[:, kpt_idx, coord] = kpt_series[np.maximum.accumulate(idx)]
        
        # Remove frames with > 30% missing joints
        valid_per_frame = np.sum(~np.isnan(imputed[:, :, 0]), axis=1)
        threshold = 0.7 * keypoints.shape[1]
        valid_frames = valid_per_frame >= threshold
        
        logger.info(f"Removed {np.sum(~valid_frames)} frames with > 30% missing joints")
        return imputed[valid_frames]
    
    def synchronize_frame_rate(
        self,
        keypoints: np.ndarray,
        original_fps: float
    ) -> np.ndarray:
        """
        Synchronize frame rate by resampling
        
        Args:
            keypoints: Array of shape (num_frames, num_keypoints, 2 or 3)
            original_fps: Original frames per second
        
        Returns:
            Resampled keypoints at target_fps
        """
        logger.info(f"Resampling from {original_fps} fps to {self.target_fps} fps...")
        
        if abs(original_fps - self.target_fps) < 0.1:
            # Already at target fps
            return keypoints
        
        # Calculate new number of frames
        duration = keypoints.shape[0] / original_fps
        new_num_frames = int(duration * self.target_fps)
        
        # Resample each keypoint coordinate
        resampled = np.zeros((new_num_frames, keypoints.shape[1], keypoints.shape[2]))
        
        for kpt_idx in range(keypoints.shape[1]):
            for coord in range(keypoints.shape[2]):
                kpt_series = keypoints[:, kpt_idx, coord]
                
                # Create interpolation function
                old_frames = np.arange(len(kpt_series))
                new_frames = np.linspace(0, len(kpt_series) - 1, new_num_frames)
                
                f = interp1d(old_frames, kpt_series, kind='linear', fill_value='extrapolate')
                resampled[:, kpt_idx, coord] = f(new_frames)
        
        logger.info(f"Frames: {keypoints.shape[0]} -> {new_num_frames}")
        return resampled
    
    def smooth_keypoints(self, keypoints: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Smooth keypoints temporally using Gaussian filter
        
        Args:
            keypoints: Array of shape (num_frames, num_keypoints, 2 or 3)
            sigma: Gaussian kernel standard deviation (in frames)
        
        Returns:
            Smoothed keypoints
        """
        logger.info(f"Smoothing keypoints with sigma={sigma}...")
        smoothed = keypoints.copy()
        
        for kpt_idx in range(keypoints.shape[1]):
            for coord in range(keypoints.shape[2]):
                smoothed[:, kpt_idx, coord] = gaussian_filter1d(
                    keypoints[:, kpt_idx, coord],
                    sigma=sigma,
                    mode='nearest'
                )
        
        return smoothed
    
    def full_pipeline(
        self,
        keypoints: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        original_fps: float = 30.0,
        normalize: bool = True,
        impute: bool = True,
        smooth: bool = True,
        resample: bool = True
    ) -> Dict:
        """
        Run full preprocessing pipeline
        
        Args:
            keypoints: Raw keypoints from pose estimator
            confidence: Confidence scores (optional)
            original_fps: Original frame rate
            normalize, impute, smooth, resample: Which steps to apply
        
        Returns:
            Dict with processed keypoints and metadata
        """
        logger.info("Starting full preprocessing pipeline...")
        
        processed = keypoints.copy()
        
        # Step 1: Normalize
        if normalize:
            processed = self.normalize_keypoints(processed)
        
        # Step 2: Handle missing keypoints
        if impute:
            processed = self.handle_missing_keypoints(processed, confidence)
        
        # Step 3: Smooth
        if smooth:
            processed = self.smooth_keypoints(processed)
        
        # Step 4: Resample to target FPS
        if resample:
            processed = self.synchronize_frame_rate(processed, original_fps)
        
        logger.info(f"Pipeline complete. Output shape: {processed.shape}")
        
        return {
            "keypoints": processed,
            "original_shape": keypoints.shape,
            "processed_shape": processed.shape,
            "num_keypoints": keypoints.shape[1],
            "original_fps": original_fps,
            "target_fps": self.target_fps
        }
    
    @staticmethod
    def _kalman_filter_1d(series: np.ndarray, process_variance: float = 1e-5,
                          measurement_variance: float = 1e-4) -> np.ndarray:
        """Simple 1D Kalman filter"""
        filtered = np.zeros_like(series)
        
        # Initialize
        x = series[0] if not np.isnan(series[0]) else 0
        p = 1.0
        
        for i, z in enumerate(series):
            if np.isnan(z):
                # Predict
                x = x
            else:
                # Update
                p = p + process_variance
                k = p / (p + measurement_variance)
                x = x + k * (z - x)
                p = (1 - k) * p
            
            filtered[i] = x
        
        return filtered


if __name__ == "__main__":
    logger.info("Keypoint preprocessor module loaded")
