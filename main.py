#!/usr/bin/env python3
"""
Main entry point for SkyNetRL - Multi-Agent Reinforcement Learning for Space-Air-Ground Networks

This script provides a unified interface to run different experiments and algorithms.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from experiments.complete_experiment import CompleteExperiment
from experiments.enhanced_complete_experiment import EnhancedCompleteExperiment


def main():
    parser = argparse.ArgumentParser(
        description="SkyNetRL: Multi-Agent RL for Space-Air-Ground Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --algorithm ae_maddpg --episodes 100
  python main.py --algorithm all --episodes 50 --comparison
  python main.py --test-mode --episodes 10
        """
    )
    
    parser.add_argument(
        "--algorithm", 
        choices=["ae_maddpg", "baseline_maddpg", "qmix", "independent_ppo", "greedy_heuristic", "random_policy", "all"],
        default="ae_maddpg",
        help="Algorithm to run (default: ae_maddpg)"
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=50,
        help="Number of training episodes (default: 50)"
    )
    
    parser.add_argument(
        "--test-episodes", 
        type=int, 
        default=10,
        help="Number of test episodes (default: 10)"
    )
    
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run comparison between all algorithms"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true", 
        help="Run in test mode with reduced episodes"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip generating visualizations"
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate video visualization of agent movements"
    )
    
    parser.add_argument(
        "--video-mode",
        choices=["overview", "tracking", "orbiting", "split_screen"],
        default="overview",
        help="Video camera mode (default: overview)"
    )
    
    parser.add_argument(
        "--realtime-3d",
        action="store_true",
        help="Enable real-time 3D visualization (requires OpenGL)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Adjust parameters for test mode
    if args.test_mode:
        args.episodes = min(args.episodes, 10)
        args.test_episodes = min(args.test_episodes, 5)
    
    # Load default configuration and update with command line args
    import json
    with open("configs/default_config.json", "r") as f:
        config = json.load(f)
    
    # Update with command line arguments
    config["seed"] = args.seed
    config["num_episodes"] = args.episodes
    config["num_eval_episodes"] = args.test_episodes
    config["enable_visualization"] = not args.no_visualization
    config["enable_video"] = args.video
    config["video_mode"] = args.video_mode
    config["enable_realtime_3d"] = args.realtime_3d
    
    # Add visualization config
    config["visualization"] = {
        "enable_video": args.video,
        "video_mode": args.video_mode,
        "enable_realtime_3d": args.realtime_3d
    }
    
    # Run experiment with enhanced visualization if video is enabled
    if args.video or args.realtime_3d:
        print("🎬 Using Enhanced Experiment with Video Visualization")
        experiment = EnhancedCompleteExperiment(config)
    else:
        print("📊 Using Standard Experiment")
        experiment = CompleteExperiment(config)
    
    if args.algorithm == "all" or args.comparison:
        print("Running complete algorithm comparison...")
        if hasattr(experiment, 'run_complete_comparison'):
            results = experiment.run_complete_comparison()
        else:
            results = experiment.run_comparison_study()
    else:
        print(f"Running single algorithm: {args.algorithm}")
        if hasattr(experiment, 'run_single_algorithm'):
            results = experiment.run_single_algorithm(args.algorithm)
        else:
            # For standard experiment, run comparison but only with specified algorithm
            config["algorithms"] = [args.algorithm]
            results = experiment.run_comparison_study()
    
    print("\nExperiment completed successfully!")
    if hasattr(experiment, 'output_dir'):
        print(f"Results saved to: {experiment.output_dir}")
    else:
        print(f"Results saved to: results/{config.get('experiment_name', 'experiment')}")


if __name__ == "__main__":
    main()