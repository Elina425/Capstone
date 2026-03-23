#!/usr/bin/env python3
"""
Quick model performance test for exercise quality assessment
"""

import cv2
import numpy as np
from pose_estimators import MediaPipeBlazePose, YOLOv11Pose
import time
import os

def test_models():
    # Find a sample video
    video_dir = "videos_ohp"
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][:1]

    if not video_files:
        print("No video files found")
        return

    video_path = os.path.join(video_dir, video_files[0])
    print(f"Testing on: {video_path}")

    # Load one frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read video frame")
        return

    print(f"Frame shape: {frame.shape}")

    results = {}

    # Test MediaPipe
    try:
        print("Testing MediaPipe BlazePose...")
        mp_model = MediaPipeBlazePose(device='cpu')
        start = time.time()
        keypoints_mp = mp_model.predict(frame)
        mp_time = time.time() - start
        results['MediaPipe'] = {
            'time': mp_time,
            'keypoints_shape': keypoints_mp.shape if keypoints_mp is not None else None,
            'success': keypoints_mp is not None
        }
        print(".3f")
    except Exception as e:
        print(f"MediaPipe failed: {e}")

    # Test YOLO
    try:
        print("Testing YOLO...")
        yolo_model = YOLOv11Pose(device='cpu')
        start = time.time()
        keypoints_yolo = yolo_model.predict(frame)
        yolo_time = time.time() - start
        results['YOLO'] = {
            'time': yolo_time,
            'keypoints_shape': keypoints_yolo.shape if keypoints_yolo is not None else None,
            'success': keypoints_yolo is not None
        }
        print(".3f")
    except Exception as e:
        print(f"YOLO failed: {e}")

    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    for model, data in results.items():
        print(f"{model}: {data['time']:.3f}s, Success: {data['success']}")

    return results

if __name__ == "__main__":
    test_models()