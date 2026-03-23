"""
Model Comparison Script
Compare pose estimation models on real project data
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

from pose_estimators import YOLOv11Pose, MediaPipeBlazePose, ViTPose, ModelComparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExerciseModelComparison:
    """Compare pose models on exercise videos"""
    
    def __init__(self, capstone_root: str, num_samples: int = 10):
        self.capstone_root = Path(capstone_root)
        self.num_samples = num_samples
        self.results = []
        
    def get_sample_videos(self, exercise_type: str) -> List[str]:
        """
        Get random sample of videos for testing
        
        Args:
            exercise_type: 'ohp' or 'squat'
        
        Returns:
            List of video paths
        """
        if exercise_type == "ohp":
            video_dir = self.capstone_root / "videos_ohp"
        elif exercise_type == "squat":
            video_dir = self.capstone_root / "videos_squat"
        else:
            raise ValueError("exercise_type must be 'ohp' or 'squat'")
        
        all_videos = list(video_dir.glob("*.mp4"))
        sample = random.sample(all_videos, min(self.num_samples, len(all_videos)))
        
        logger.info(f"Selected {len(sample)} sample videos for {exercise_type.upper()}")
        return [str(v) for v in sample]
    
    def compare_on_exercise(self, exercise_type: str, device: str = "cuda") -> pd.DataFrame:
        """
        Compare models on exercise videos
        
        Args:
            exercise_type: 'ohp' or 'squat'
            device: 'cuda' or 'cpu'
        
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"="*80)
        logger.info(f"COMPARING MODELS ON {exercise_type.upper()}")
        logger.info(f"="*80)
        
        # Get sample videos
        videos = self.get_sample_videos(exercise_type)
        
        # Initialize models
        models_to_compare = [
            YOLOv11Pose(model_size="m", device=device),
            MediaPipeBlazePose(device=device),
            ViTPose(model_type="base", device=device, use_dora=True),
        ]
        
        results = []
        
        # Test each video
        for i, video_path in enumerate(videos, 1):
            logger.info(f"\n[{i}/{len(videos)}] Testing: {Path(video_path).name}")
            
            # Create comparison
            comparison = ModelComparison(video_path, models_to_compare)
            
            # Run with limited frames for faster comparison
            video_results = comparison.run_comparison(max_frames=100)
            
            # Log summary
            logger.info(comparison.get_summary())
            
            # Store results
            for model_name, metrics in video_results.items():
                result_row = {
                    "exercise": exercise_type,
                    "video_id": Path(video_path).stem,
                    "model": model_name,
                    **metrics
                }
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def run_full_comparison(self, device: str = "cuda") -> Dict:
        """
        Run full comparison on both exercises
        
        Args:
            device: 'cuda' or 'cpu'
        
        Returns:
            Dictionary with results for both exercises
        """
        logger.info("\n" + "="*80)
        logger.info("FULL MODEL COMPARISON - STARTING")
        logger.info("="*80 + "\n")
        
        all_results = {}
        
        for exercise_type in ["ohp", "squat"]:
            try:
                results_df = self.compare_on_exercise(exercise_type, device)
                all_results[exercise_type] = results_df
                
                # Print summary statistics
                logger.info(f"\n{exercise_type.upper()} Summary Statistics:")
                summary = results_df.groupby("model").agg({
                    "avg_inference_fps": ["mean", "std"],
                    "total_time": ["mean", "std"],
                    "success_rate": ["mean", "min"]
                }).round(2)
                logger.info(f"\n{summary}")
                
            except Exception as e:
                logger.error(f"Error comparing on {exercise_type}: {e}")
        
        return all_results
    
    def save_results(self, results: Dict, output_dir: str):
        """Save comparison results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV files
        for exercise_type, df in results.items():
            csv_path = output_dir / f"{exercise_type}_model_comparison.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved: {csv_path}")
        
        # Save comprehensive JSON report
        report = {}
        for exercise_type, df in results.items():
            report[exercise_type] = {
                "total_videos_tested": len(df['video_id'].unique()),
                "models_tested": df['model'].unique().tolist(),
                "summary_by_model": df.groupby("model").agg({
                    "avg_inference_fps": ["mean", "std", "min", "max"],
                    "total_time": ["mean", "std"],
                    "success_rate": ["mean", "min"]
                }).to_dict()
            }
        
        json_path = output_dir / "comparison_report.json"
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved: {json_path}")
    
    def generate_recommendations(self, results: Dict) -> str:
        """
        Generate model selection recommendations based on comparison
        
        Returns:
            Recommendation report as string
        """
        report = "\n" + "="*80 + "\n"
        report += "MODEL SELECTION RECOMMENDATIONS\n"
        report += "="*80 + "\n\n"
        
        for exercise_type, df in results.items():
            report += f"\n{exercise_type.upper()} EXERCISE:\n"
            report += "-"*40 + "\n"
            
            # Best overall model (highest FPS)
            best_fps = df.loc[df['avg_inference_fps'].idxmax()]
            report += f"1. FASTEST: {best_fps['model']}\n"
            report += f"   - Avg FPS: {best_fps['avg_inference_fps']:.2f}\n"
            report += f"   - Success Rate: {best_fps['success_rate']:.1%}\n"
            
            # Most reliable (highest success rate)
            best_success = df.loc[df['success_rate'].idxmax()]
            report += f"\n2. MOST RELIABLE: {best_success['model']}\n"
            report += f"   - Success Rate: {best_success['success_rate']:.1%}\n"
            report += f"   - Avg FPS: {best_success['avg_inference_fps']:.2f}\n"
            
            # Summary table
            summary = df.groupby("model")[['avg_inference_fps', 'success_rate']].mean()
            report += f"\n3. SUMMARY TABLE:\n{summary.round(2).to_string()}\n"
            
            # Recommendation
            if best_fps['avg_inference_fps'] >= 30:
                recommended = best_fps['model']
                reason = "Sufficient speed for real-time processing"
            else:
                recommended = best_success['model']
                reason = "Prioritizing accuracy over speed"
            
            report += f"\n✅ RECOMMENDED: {recommended}\n"
            report += f"   Reason: {reason}\n"
        
        report += "\n" + "="*80 + "\n"
        report += "NOTES:\n"
        report += "- Transformer models (ViTPose with DoRA) selected for state-of-the-art accuracy\n"
        report += "- MediaPipe recommended for edge deployment (fast inference)\n"
        report += "- YOLOv11 recommended for balanced speed/accuracy\n"
        report += "="*80 + "\n"
        
        return report


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare pose estimation models")
    parser.add_argument("--capstone-root", default="/Users/emelkonyan/PycharmProjects/Capstone",
                        help="Path to Capstone project root")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of videos to sample for comparison")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--output-dir", default="results/model_comparison",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = ExerciseModelComparison(args.capstone_root, args.num_samples)
    results = comparator.run_full_comparison(device=args.device)
    
    # Save results
    comparator.save_results(results, args.output_dir)
    
    # Print recommendations
    recommendations = comparator.generate_recommendations(results)
    print(recommendations)
    
    # Save recommendations
    rec_path = Path(args.output_dir) / "recommendations.txt"
    with open(rec_path, 'w') as f:
        f.write(recommendations)
    logger.info(f"Saved: {rec_path}")


if __name__ == "__main__":
    main()
