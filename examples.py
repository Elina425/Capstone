"""
Example Usage of Exercise Quality Assessment Pipeline
Shows how to use analyze_dataset, compare_models, and preprocess_keypoints
"""

import json
from pathlib import Path
import numpy as np
from analyze_dataset import DatasetAnalyzer
from pose_estimators import YOLOv11Pose, ModelComparison
from keypoint_preprocessor import KeypointPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_analyze_dataset():
    """
    Example 1: Analyze the dataset
    Verifies data integrity and generates statistics
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: DATASET ANALYSIS")
    print("="*80 + "\n")
    
    capstone_root = "/Users/emelkonyan/PycharmProjects/Capstone"
    analyzer = DatasetAnalyzer(capstone_root)
    
    # Run full analysis
    analysis = analyzer.run_full_analysis()
    
    # Save results
    output_path = Path(capstone_root) / "dataset_analysis.json"
    analyzer.save_analysis(analysis, str(output_path))
    
    # Print summary
    analyzer.print_summary(analysis)
    
    # Access specific results
    ohp_data = analysis["ohp"]
    squat_data = analysis["squat"]
    
    print(f"\nOHP Total Videos: {ohp_data['total_videos']}")
    print(f"Squat Total Videos: {squat_data['total_videos']}")
    
    return analysis


def example_2_single_model_inference():
    """
    Example 2: Run single model inference on a video
    Extract keypoints using one model
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: SINGLE MODEL INFERENCE")
    print("="*80 + "\n")
    
    capstone_root = "/Users/emelkonyan/PycharmProjects/Capstone"
    video_path = str(Path(capstone_root) / "videos_ohp" / "62805_6.mp4")
    
    # Create YOLOv11 model
    try:
        model = YOLOv11Pose(model_size="m", device="cuda")
        
        # Process video (limit to 50 frames for speed)
        result = model.process_video(video_path, max_frames=50)
        
        if result:
            print(f"\n✅ Inference successful!")
            print(f"  Frames processed: {result['num_frames']}")
            print(f"  Output shape: {result['keypoints'].shape}")
            print(f"  Keypoints per frame: {result['num_keypoints']}")
            print(f"  Avg FPS: {result['avg_inference_fps']:.2f}")
            print(f"  Total time: {result['total_time']:.2f}s")
            
            # Show sample keypoints
            sample_frame = result['keypoints'][0]
            print(f"\nSample frame keypoints (frame 0):")
            print(f"  Shape: {sample_frame.shape}")
            print(f"  First 5 keypoints:\n{sample_frame[:5]}")
        else:
            print("❌ Inference failed")
    
    except ImportError as e:
        print(f"⚠️  Model not available: {e}")
        print("    Install with: pip install ultralytics")
    
    return result


def example_3_preprocess_keypoints():
    """
    Example 3: Preprocess extracted keypoints
    Apply normalization, imputation, and frame rate sync
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: KEYPOINT PREPROCESSING")
    print("="*80 + "\n")
    
    # Create sample keypoints (simulate extraction)
    num_frames = 100
    num_keypoints = 17
    sample_keypoints = np.random.randn(num_frames, num_keypoints, 2) * 100 + 200
    sample_confidence = np.random.rand(num_frames, num_keypoints)
    
    print(f"Input shape: {sample_keypoints.shape}")
    print(f"Input FPS: 30")
    
    # Create preprocessor
    preprocessor = KeypointPreprocessor(target_fps=30, confidence_threshold=0.3)
    
    # Run full pipeline
    result = preprocessor.full_pipeline(
        keypoints=sample_keypoints,
        confidence=sample_confidence,
        original_fps=30.0,
        normalize=True,
        impute=True,
        smooth=True,
        resample=True
    )
    
    print(f"\n✅ Preprocessing complete!")
    print(f"  Original shape: {result['original_shape']}")
    print(f"  Processed shape: {result['processed_shape']}")
    print(f"  Frame reduction: {result['original_shape'][0]} → {result['processed_shape'][0]}")
    
    processed_kpts = result['keypoints']
    print(f"\nProcessed keypoints statistics:")
    print(f"  Min: {processed_kpts.min():.3f}")
    print(f"  Max: {processed_kpts.max():.3f}")
    print(f"  Mean: {processed_kpts.mean():.3f}")
    print(f"  Std: {processed_kpts.std():.3f}")
    
    return result


def example_4_extract_and_preprocess():
    """
    Example 4: End-to-end - Extract keypoints and preprocess
    Complete pipeline from video to processed keypoints
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: END-TO-END PIPELINE (Extract + Preprocess)")
    print("="*80 + "\n")
    
    capstone_root = "/Users/emelkonyan/PycharmProjects/Capstone"
    video_path = str(Path(capstone_root) / "videos_ohp" / "62805_6.mp4")
    
    try:
        # Step 1: Extract keypoints
        print("Step 1: Extracting keypoints...")
        model = YOLOv11Pose(model_size="m", device="cuda")
        extraction_result = model.process_video(video_path, max_frames=100)
        
        if extraction_result is None:
            print("❌ Extraction failed")
            return None
        
        raw_keypoints = extraction_result['keypoints']
        print(f"  ✅ Extracted {raw_keypoints.shape[0]} frames")
        print(f"  Shape: {raw_keypoints.shape}")
        
        # Step 2: Preprocess
        print("\nStep 2: Preprocessing keypoints...")
        preprocessor = KeypointPreprocessor(target_fps=30)
        
        preprocess_result = preprocessor.full_pipeline(
            keypoints=raw_keypoints,
            original_fps=extraction_result['fps'],
            normalize=True,
            impute=True,
            smooth=True,
            resample=True
        )
        
        processed_keypoints = preprocess_result['keypoints']
        print(f"  ✅ Preprocessing complete")
        print(f"  Output shape: {processed_keypoints.shape}")
        
        # Summary
        print("\n" + "-"*80)
        print("PIPELINE SUMMARY:")
        print(f"  Raw keypoints shape: {raw_keypoints.shape}")
        print(f"  Processed keypoints shape: {processed_keypoints.shape}")
        print(f"  Inference FPS: {extraction_result['avg_inference_fps']:.2f}")
        print(f"  Video FPS: {extraction_result['fps']}")
        print(f"  Target FPS: {preprocessor.target_fps}")
        
        return {
            "raw": raw_keypoints,
            "processed": processed_keypoints,
            "metadata": preprocess_result
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def example_5_compare_models():
    """
    Example 5: Compare models on same video
    See which model works best for your data
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: MODEL COMPARISON")
    print("="*80 + "\n")
    
    capstone_root = "/Users/emelkonyan/PycharmProjects/Capstone"
    video_path = str(Path(capstone_root) / "videos_ohp" / "62805_6.mp4")
    
    print(f"Testing on: {Path(video_path).name}\n")
    
    # Create models to compare
    models_to_compare = [
        YOLOv11Pose(model_size="m", device="cuda"),
        # MediaPipeBlazePose(device="cuda"),  # Uncomment if installed
    ]
    
    # Compare
    comparison = ModelComparison(video_path, models_to_compare)
    results = comparison.run_comparison(max_frames=50)
    
    # Print summary
    print(comparison.get_summary())
    
    # Return structured results
    results_dict = {}
    for model_name, metrics in results.items():
        results_dict[model_name] = metrics
    
    return results_dict


def example_6_batch_processing():
    """
    Example 6: Batch process multiple videos
    Process several videos and save results
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: BATCH PROCESSING")
    print("="*80 + "\n")
    
    capstone_root = Path("/Users/emelkonyan/PycharmProjects/Capstone")
    video_dir = capstone_root / "videos_ohp"
    output_dir = capstone_root / "keypoints" / "ohp_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first 3 videos
    videos = list(video_dir.glob("*.mp4"))[:3]
    print(f"Processing {len(videos)} videos...\n")
    
    try:
        model = YOLOv11Pose(model_size="m", device="cuda")
        
        results = {}
        for i, video_path in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] Processing: {video_path.name}")
            
            result = model.process_video(str(video_path), max_frames=50)
            if result:
                video_id = video_path.stem
                results[video_id] = {
                    "num_frames": result['num_frames'],
                    "fps": result['fps'],
                    "inference_fps": result['avg_inference_fps']
                }
                print(f"  ✅ {result['num_frames']} frames, {result['avg_inference_fps']:.2f} fps\n")
            else:
                print(f"  ❌ Failed\n")
        
        # Save batch results
        batch_summary = {
            "total_videos": len(videos),
            "successful": len(results),
            "results": results
        }
        
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Results saved to: {summary_file}")
        
        return batch_summary
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("EXERCISE QUALITY ASSESSMENT - USAGE EXAMPLES")
    print("="*80)
    
    # Run examples
    examples = [
        ("Dataset Analysis", example_1_analyze_dataset),
        ("Single Model Inference", example_2_single_model_inference),
        ("Keypoint Preprocessing", example_3_preprocess_keypoints),
        ("End-to-End Pipeline", example_4_extract_and_preprocess),
        ("Model Comparison", example_5_compare_models),
        ("Batch Processing", example_6_batch_processing),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning Examples 1-3 (fastest)...\n")
    
    try:
        example_1_analyze_dataset()
    except Exception as e:
        logger.error(f"Example 1 error: {e}")
    
    try:
        example_3_preprocess_keypoints()
    except Exception as e:
        logger.error(f"Example 3 error: {e}")
    
    print("\n" + "="*80)
    print("Note: Examples 2,4,5,6 require GPU and model downloads")
    print("Run individually as needed with proper GPU setup")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
