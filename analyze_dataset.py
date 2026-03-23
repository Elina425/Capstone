"""
Dataset Analysis and Verification Script
Analyzes the exercise videos, labels, and train/test splits
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze and verify the exercise dataset"""
    
    def __init__(self, capstone_root: str):
        self.root = Path(capstone_root)
        self.videos_ohp = self.root / "videos_ohp"
        self.videos_squat = self.root / "videos_squat"
        self.labels_ohp = self.root / "Labels_ohp"
        self.labels_squat = self.root / "errors_squat"
        self.eval_ohp = self.root / "ohp_eval"
        self.eval_squat = self.root / "squat_eval"
        
    def get_video_files(self, exercise_type: str) -> dict:
        """Get all video files for an exercise type"""
        if exercise_type == "ohp":
            video_dir = self.videos_ohp
        elif exercise_type == "squat":
            video_dir = self.videos_squat
        else:
            raise ValueError("exercise_type must be 'ohp' or 'squat'")
        
        videos = {}
        if video_dir.exists():
            for video_file in video_dir.glob("*.mp4"):
                # Extract video_id from filename (e.g., "62794_6.mp4" -> "62794_6")
                video_id = video_file.stem
                videos[video_id] = str(video_file)
        
        return videos
    
    def load_labels(self, exercise_type: str) -> dict:
        """Load error labels for an exercise type"""
        if exercise_type == "ohp":
            label_dir = self.labels_ohp
            label_files = {
                "error_elbows": label_dir / "error_elbows.json",
                "error_knees": label_dir / "error_knees.json"
            }
        elif exercise_type == "squat":
            label_dir = self.labels_squat
            label_files = {
                "error_knees_forward": label_dir / "error_knees_forward.json",
                "error_knees_inward": label_dir / "error_knees_inward.json"
            }
        else:
            raise ValueError("exercise_type must be 'ohp' or 'squat'")
        
        labels = {}
        for error_type, label_file in label_files.items():
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels[error_type] = json.load(f)
            else:
                logger.warning(f"Label file not found: {label_file}")
                labels[error_type] = {}
        
        return labels
    
    def load_splits(self, exercise_type: str) -> dict:
        """Load train/val/test split indices"""
        if exercise_type == "ohp":
            eval_dir = self.eval_ohp
        elif exercise_type == "squat":
            eval_dir = self.eval_squat
        else:
            raise ValueError("exercise_type must be 'ohp' or 'squat'")
        
        splits = {}
        split_files = {
            "train": eval_dir / "train_keys.json",
            "val": eval_dir / "val_keys.json",
            "test": eval_dir / "test_keys.json"
        }
        
        for split_name, split_file in split_files.items():
            if split_file.exists():
                with open(split_file, 'r') as f:
                    splits[split_name] = json.load(f)
            else:
                logger.warning(f"Split file not found: {split_file}")
                splits[split_name] = []
        
        return splits
    
    def analyze_exercise(self, exercise_type: str) -> dict:
        """Comprehensive analysis of one exercise type"""
        logger.info(f"Analyzing {exercise_type.upper()} dataset...")
        
        videos = self.get_video_files(exercise_type)
        labels = self.load_labels(exercise_type)
        splits = self.load_splits(exercise_type)
        
        # Calculate statistics
        num_videos = len(videos)
        
        # Count labeled videos and error occurrences
        error_stats = {}
        for error_type, error_labels in labels.items():
            videos_with_errors = sum(1 for v_id, errors in error_labels.items() if errors)
            total_error_instances = sum(len(errors) for errors in error_labels.values())
            error_stats[error_type] = {
                "videos_with_errors": videos_with_errors,
                "total_error_instances": total_error_instances,
                "error_prevalence": videos_with_errors / max(len(error_labels), 1)
            }
        
        # Verify split distribution
        split_stats = {}
        all_split_videos = set()
        for split_name, split_ids in splits.items():
            split_stats[split_name] = len(split_ids)
            all_split_videos.update(set(split_ids))
        
        # Check for overlaps and missing videos
        overlap_stats = {
            "train_test_overlap": len(set(splits.get("train", [])) & set(splits.get("test", []))),
            "train_val_overlap": len(set(splits.get("train", [])) & set(splits.get("val", []))),
            "val_test_overlap": len(set(splits.get("val", [])) & set(splits.get("test", [])))
        }
        
        missing_in_splits = set(videos.keys()) - all_split_videos
        extra_in_splits = all_split_videos - set(videos.keys())
        
        return {
            "exercise_type": exercise_type,
            "total_videos": num_videos,
            "error_statistics": error_stats,
            "split_distribution": split_stats,
            "data_integrity": {
                "videos_not_in_splits": len(missing_in_splits),
                "split_videos_not_found": len(extra_in_splits),
                "missing_videos": list(missing_in_splits)[:10],  # Show first 10
                "extra_videos": list(extra_in_splits)[:10],  # Show first 10
                "split_overlaps": overlap_stats
            }
        }
    
    def run_full_analysis(self) -> dict:
        """Run complete dataset analysis"""
        analysis = {
            "timestamp": str(os.popen('date').read()),
            "ohp": self.analyze_exercise("ohp"),
            "squat": self.analyze_exercise("squat")
        }
        
        return analysis
    
    def save_analysis(self, analysis: dict, output_path: str):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to {output_path}")
    
    def print_summary(self, analysis: dict):
        """Print human-readable summary"""
        print("\n" + "="*80)
        print("DATASET ANALYSIS SUMMARY")
        print("="*80)
        
        for exercise_type in ["ohp", "squat"]:
            data = analysis[exercise_type]
            print(f"\n{exercise_type.upper()} EXERCISE:")
            print(f"  Total Videos: {data['total_videos']}")
            print(f"  Split Distribution: Train={data['split_distribution']['train']}, "
                  f"Val={data['split_distribution']['val']}, "
                  f"Test={data['split_distribution']['test']}")
            
            print(f"  Error Types:")
            for error_type, stats in data['error_statistics'].items():
                print(f"    - {error_type}:")
                print(f"        Videos with errors: {stats['videos_with_errors']}")
                print(f"        Total error instances: {stats['total_error_instances']}")
                print(f"        Prevalence: {stats['error_prevalence']:.2%}")
            
            integrity = data['data_integrity']
            print(f"  Data Integrity:")
            print(f"    Videos not in splits: {integrity['videos_not_in_splits']}")
            print(f"    Split videos not found: {integrity['split_videos_not_found']}")
            
            if integrity['split_overlaps']['train_test_overlap'] > 0:
                print(f"  ⚠️  WARNING: Train-Test overlap: {integrity['split_overlaps']['train_test_overlap']}")
            if integrity['split_overlaps']['train_val_overlap'] > 0:
                print(f"  ⚠️  WARNING: Train-Val overlap: {integrity['split_overlaps']['train_val_overlap']}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    capstone_root = "/Users/emelkonyan/PycharmProjects/Capstone"
    
    # Create analyzer
    analyzer = DatasetAnalyzer(capstone_root)
    
    # Run analysis
    analysis = analyzer.run_full_analysis()
    
    # Save results
    output_path = os.path.join(capstone_root, "dataset_analysis.json")
    analyzer.save_analysis(analysis, output_path)
    
    # Print summary
    analyzer.print_summary(analysis)
    
    print(f"\n✅ Analysis complete. Results saved to {output_path}")
