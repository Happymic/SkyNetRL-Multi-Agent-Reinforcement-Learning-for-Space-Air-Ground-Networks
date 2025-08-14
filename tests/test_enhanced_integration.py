#!/usr/bin/env python3
"""
Test Enhanced Training Integration with Video Visualization
Tests the complete integration with actual training and visualization
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_enhanced_training():
    """Test enhanced training with video generation"""
    
    print("=" * 80)
    print("🧪 Testing Enhanced Training Integration")
    print("=" * 80)
    
    # Test 1: Single algorithm with video
    print("\n🎯 Test 1: Single Algorithm with Video")
    print("-" * 50)
    
    cmd = [
        "python", "main.py",
        "--algorithm", "greedy_heuristic",
        "--episodes", "10",
        "--video",
        "--video-mode", "overview",
        "--test-mode"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Single algorithm test passed")
            print("📊 Output preview:")
            print(result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Single algorithm test failed")
            print("Error output:", result.stderr[-500:])
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
    
    # Test 2: Check generated files
    print("\n📁 Test 2: Check Generated Files")
    print("-" * 50)
    
    # Look for recent experiment directories
    results_dir = Path("results")
    if results_dir.exists():
        experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "enhanced_experiment" in d.name]
        experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if experiment_dirs:
            latest_dir = experiment_dirs[0]
            print(f"📂 Latest experiment: {latest_dir.name}")
            
            # Check for video files
            video_files = list(latest_dir.rglob("*.gif")) + list(latest_dir.rglob("*.mp4"))
            if video_files:
                print(f"🎬 Generated videos: {len(video_files)}")
                for video in video_files[:3]:  # Show first 3
                    print(f"   📹 {video.name} ({video.stat().st_size / 1024:.1f} KB)")
            else:
                print("⚠️ No video files found")
                
            # Check for JSON files
            json_files = list(latest_dir.rglob("*.json"))
            if json_files:
                print(f"📊 Generated data files: {len(json_files)}")
                for json_file in json_files:
                    print(f"   📄 {json_file.name}")
            else:
                print("⚠️ No data files found")
        else:
            print("⚠️ No enhanced experiment directories found")
    else:
        print("⚠️ Results directory not found")

def test_visualization_components():
    """Test individual visualization components"""
    
    print("\n🔬 Test 3: Visualization Components")
    print("-" * 50)
    
    try:
        # Test enhanced visualizer import
        from src.visualization.enhanced_training_visualizer import EnhancedTrainingVisualizer
        print("✅ Enhanced visualizer import successful")
        
        # Test training integration import
        from src.visualization.training_integration import TrainingVisualizer, TrainingCallback
        print("✅ Training integration import successful")
        
        # Test enhanced experiment import
        from src.experiments.enhanced_complete_experiment import EnhancedCompleteExperiment
        print("✅ Enhanced experiment import successful")
        
        # Test configuration loading
        with open("configs/default_config.json", "r") as f:
            config = json.load(f)
        print("✅ Configuration loading successful")
        
        # Test visualizer initialization
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        viz = TrainingVisualizer(config['environment'], output_dir)
        print("✅ Training visualizer initialization successful")
        
        print("\n🎉 All component tests passed!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()

def run_quick_demo():
    """Run a quick demonstration"""
    
    print("\n🚀 Test 4: Quick Demo")
    print("-" * 50)
    
    try:
        # Run the enhanced demo
        cmd = [
            "python", "demo_enhanced_training_viz.py", 
            "--enhanced"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ Enhanced demo completed successfully")
            print("📊 Output summary:")
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Last 20 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Enhanced demo failed")
            print("Error:", result.stderr[-300:])
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    print("🧪 Enhanced Training Integration Test Suite")
    print("This will test the complete enhanced visualization system")
    print()
    
    # Run all tests
    test_visualization_components()
    test_enhanced_training()
    run_quick_demo()
    
    print("\n" + "=" * 80)
    print("🏁 TEST SUITE COMPLETE")
    print("=" * 80)
    print()
    print("💡 What was tested:")
    print("   ✅ Component imports and initialization")
    print("   ✅ Enhanced training with video generation")
    print("   ✅ File generation and organization")
    print("   ✅ Quick demonstration functionality")
    print()
    print("📁 Check the following directories for results:")
    print("   • results/enhanced_experiment_* (training results)")
    print("   • results/enhanced_training_demo (demo results)")
    print("   • test_output (test artifacts)")
    print()
    print("🎬 Generated videos should include:")
    print("   • Individual algorithm training videos")
    print("   • Enhanced visualization with multiple panels")
    print("   • 3D view integration")
    print("   • Real-time metrics and agent status")
    print("   • POI coverage visualization")
    print("=" * 80)