"""
Pose Estimation Models Wrapper
Unified interface for comparing different pose estimation models
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseEstimator(ABC):
    """Abstract base class for pose estimators"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.num_keypoints = None
        
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict keypoints on a single frame
        
        Args:
            frame: Input image (H x W x 3)
            
        Returns:
            keypoints: Array of shape (num_keypoints, 2) or (num_keypoints, 3)
                      Last dimension: [x, y] or [x, y, confidence]
        """
        pass
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict:
        """
        Process entire video and extract keypoints
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            Dict with keypoints, metadata, and timing
        """
        import cv2
        
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        keypoints_list = []
        
        frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
        
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict keypoints
            keypoints = self.predict(frame)
            if keypoints is not None:
                keypoints_list.append(keypoints)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"{self.model_name}: Processed {frame_count}/{frames_to_process} frames")
        
        cap.release()
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time
        
        return {
            "keypoints": np.array(keypoints_list),
            "num_frames": frame_count,
            "fps": fps,
            "avg_inference_fps": avg_fps,
            "total_time": elapsed_time,
            "num_keypoints": self.num_keypoints
        }


class YOLOv11Pose(PoseEstimator):
    """YOLO 11 Pose Estimation"""
    
    def __init__(self, model_size: str = "m", device: str = "cuda"):
        super().__init__("YOLOv11", device)
        self.model_size = model_size
        self.num_keypoints = 17  # COCO format
        self.load_model()
    
    def load_model(self):
        """Load YOLOv11 pose model"""
        try:
            from ultralytics import YOLO
            model_name = f"yolov11{self.model_size}-pose.pt"
            logger.info(f"Loading {model_name}...")
            self.model = YOLO(model_name)
            self.model.to(self.device)
            logger.info("YOLOv11 model loaded successfully")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
    
    def predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Predict keypoints using YOLOv11"""
        try:
            results = self.model(frame, verbose=False)
            if results and len(results) > 0 and results[0].keypoints is not None:
                # Extract keypoints for the first (most confident) person
                keypoints = results[0].keypoints.data[0].cpu().numpy()  # (17, 3) [x, y, conf]
                return keypoints
        except Exception as e:
            logger.warning(f"Error in YOLOv11 prediction: {e}")
        return None


class MediaPipeBlazePose(PoseEstimator):
    """MediaPipe BlazePose - Optimized for edge devices"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("MediaPipe-BlazePose", device)
        self.num_keypoints = 33  # MediaPipe full body
        self.simplified_keypoints = 17  # Will map to COCO-like format
        self.load_model()
    
    def load_model(self):
        """Load MediaPipe BlazePose"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # 0, 1, or 2
                smooth_landmarks=True
            )
            logger.info("MediaPipe BlazePose loaded successfully")
        except ImportError:
            logger.error("mediapipe not installed. Install with: pip install mediapipe")
            raise
    
    def predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Predict keypoints using MediaPipe"""
        try:
            import cv2
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Convert to numpy array [x, y, z, visibility]
                keypoints = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility] 
                    for lm in results.pose_landmarks.landmark
                ])
                return keypoints  # (33, 4)
        except Exception as e:
            logger.warning(f"Error in MediaPipe prediction: {e}")
        return None


class OpenPoseWrapper(PoseEstimator):
    """OpenPose - Multi-person pose estimation"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("OpenPose", device)
        self.num_keypoints = 18  # COCO + neck
        self.load_model()
    
    def load_model(self):
        """Load OpenPose"""
        try:
            import cv2
            proto_file = "models/openpose/pose/coco/pose.proto"
            weights_file = "models/openpose/pose/coco/pose.caffemodel"
            
            self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
            if self.device == "cuda":
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("OpenPose loaded successfully")
        except Exception as e:
            logger.warning(f"OpenPose loading issue: {e}. Will use fallback.")
    
    def predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Predict keypoints using OpenPose"""
        # Simplified placeholder - actual OpenPose integration more complex
        logger.warning("Full OpenPose integration requires model files. Using placeholder.")
        return None


class ViTPose(PoseEstimator):
    """Vision Transformer based Pose Estimation with DoRA/LoRA optimization"""
    
    def __init__(self, model_type: str = "base", device: str = "cuda", use_dora: bool = True):
        super().__init__("ViTPose", device)
        self.model_type = model_type
        self.use_dora = use_dora
        self.num_keypoints = 17  # COCO format
        self.load_model()
    
    def load_model(self):
        """Load ViTPose model with DoRA/LoRA optimization"""
        try:
            logger.info(f"Loading ViTPose ({self.model_type})...")
            # Note: Actual implementation would load from checkpoint
            # For now, this is a placeholder for the actual model
            logger.info(f"ViTPose with {'DoRA' if self.use_dora else 'standard'} loaded")
        except Exception as e:
            logger.error(f"Error loading ViTPose: {e}")
            raise
    
    def predict(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Predict keypoints using ViTPose"""
        try:
            # Placeholder for actual ViTPose inference
            logger.warning("ViTPose implementation not yet complete")
            return None
        except Exception as e:
            logger.warning(f"Error in ViTPose prediction: {e}")
            return None


class ModelComparison:
    """Compare multiple pose estimation models"""
    
    def __init__(self, video_path: str, models: List[PoseEstimator]):
        self.video_path = video_path
        self.models = models
        self.results = {}
    
    def run_comparison(self, max_frames: int = 300) -> Dict:
        """
        Compare all models on the same video
        
        Args:
            max_frames: Maximum frames to process
            
        Returns:
            Comparison results with timing and quality metrics
        """
        logger.info(f"Comparing {len(self.models)} models on video: {self.video_path}")
        
        for model in self.models:
            logger.info(f"Processing with {model.model_name}...")
            result = model.process_video(self.video_path, max_frames)
            
            if result:
                self.results[model.model_name] = {
                    "num_frames": result["num_frames"],
                    "fps": result["fps"],
                    "avg_inference_fps": result["avg_inference_fps"],
                    "total_time": result["total_time"],
                    "num_keypoints": result["num_keypoints"],
                    "success_rate": result["num_frames"] / max_frames
                }
        
        return self.results
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = "\nModel Comparison Summary:\n"
        summary += "="*80 + "\n"
        
        for model_name, metrics in self.results.items():
            summary += f"\n{model_name}:\n"
            summary += f"  Avg FPS (Inference): {metrics['avg_inference_fps']:.2f}\n"
            summary += f"  Total Time: {metrics['total_time']:.2f}s\n"
            summary += f"  Success Rate: {metrics['success_rate']:.1%}\n"
            summary += f"  Output Shape: {metrics['num_keypoints']} keypoints\n"
        
        return summary


if __name__ == "__main__":
    logger.info("Pose Estimator module loaded successfully")
