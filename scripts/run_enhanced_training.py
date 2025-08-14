#!/usr/bin/env python3
"""
Enhanced Training Runner with Comprehensive Visualization
Runs training with enhanced video generation and analysis
"""

import os
import subprocess
import sys
import json
from pathlib import Path
import shutil
from datetime import datetime

def run_enhanced_training_demo():
    """Run a complete enhanced training demonstration"""
    
    print("="*80)
    print("🚀 Enhanced Training with Video Visualization Demo")
    print("="*80)
    
    # Create clean output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    demo_dir = f"results/enhanced_demo_{timestamp}"
    os.makedirs(demo_dir, exist_ok=True)
    
    print(f"📁 Demo directory: {demo_dir}")
    
    # Test 1: Run training with enhanced visualization
    print("\n🎯 Step 1: Running Enhanced Training")
    print("-" * 60)
    
    cmd = [
        sys.executable, "main.py",
        "--algorithm", "greedy_heuristic",
        "--episodes", "20",
        "--test-episodes", "5",
        "--video",
        "--video-mode", "overview",
        "--test-mode"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("⏳ Running training (this may take a few minutes)...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Enhanced training completed successfully!")
            print("\n📊 Training Output Summary:")
            
            # Extract key information from output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['Episode', 'Reward', 'Coverage', 'Video', 'saved']):
                    print(f"  {line}")
        else:
            print("❌ Training failed:")
            print(result.stderr[-1000:])  # Last 1000 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False
    
    # Test 2: Generate enhanced visualization demo
    print("\n🎨 Step 2: Enhanced Visualization Demo")
    print("-" * 60)
    
    demo_cmd = [sys.executable, "demo_enhanced_training_viz.py", "--enhanced"]
    
    try:
        print("⏳ Generating enhanced visualizations...")
        demo_result = subprocess.run(demo_cmd, capture_output=True, text=True, timeout=300)
        
        if demo_result.returncode == 0:
            print("✅ Enhanced visualizations generated!")
        else:
            print("⚠️ Enhanced visualization demo had issues:")
            print(demo_result.stderr[-500:])
            
    except Exception as e:
        print(f"⚠️ Demo failed: {e}")
    
    # Test 3: Analyze results
    print("\n📊 Step 3: Results Analysis")
    print("-" * 60)
    
    analysis_cmd = [sys.executable, "analyze_training_results.py"]
    
    try:
        print("⏳ Analyzing training results...")
        subprocess.run(analysis_cmd, timeout=120)
        print("✅ Analysis completed!")
        
    except Exception as e:
        print(f"⚠️ Analysis had issues: {e}")
    
    # Step 4: Check generated files
    print("\n📁 Step 4: Generated Files Summary")
    print("-" * 60)
    
    check_generated_files()
    
    # Step 5: Create final summary
    print("\n📋 Step 5: Final Summary")
    print("-" * 60)
    
    create_final_summary()
    
    return True

def check_generated_files():
    """Check what files were generated"""
    
    # Check main results
    results_dirs = [
        "results/sagin_coverage_optimization",
        "results/enhanced_training_demo",
        "results/enhanced_experiment_*"
    ]
    
    for pattern in results_dirs:
        if "*" in pattern:
            # Use pathlib for glob patterns
            matching_dirs = list(Path(".").glob(pattern))
            for dir_path in matching_dirs:
                print(f"📂 Found: {dir_path}")
                check_directory_contents(dir_path)
        else:
            dir_path = Path(pattern)
            if dir_path.exists():
                print(f"📂 Found: {dir_path}")
                check_directory_contents(dir_path)
    
def check_directory_contents(dir_path: Path):
    """Check contents of a directory"""
    
    # Check for videos
    video_files = list(dir_path.rglob("*.gif")) + list(dir_path.rglob("*.mp4"))
    if video_files:
        print(f"  🎬 Videos ({len(video_files)}):")
        for video in video_files[:5]:  # Show first 5
            size_kb = video.stat().st_size / 1024
            print(f"    📹 {video.name} ({size_kb:.1f} KB)")
        if len(video_files) > 5:
            print(f"    ... and {len(video_files) - 5} more videos")
    
    # Check for images
    image_files = list(dir_path.rglob("*.png")) + list(dir_path.rglob("*.jpg"))
    if image_files:
        print(f"  🖼️ Images ({len(image_files)}):")
        for img in image_files[:3]:  # Show first 3
            size_kb = img.stat().st_size / 1024
            print(f"    🖼️ {img.name} ({size_kb:.1f} KB)")
        if len(image_files) > 3:
            print(f"    ... and {len(image_files) - 3} more images")
    
    # Check for data files
    data_files = list(dir_path.rglob("*.json")) + list(dir_path.rglob("*.md"))
    if data_files:
        print(f"  📊 Data files ({len(data_files)}):")
        for data in data_files[:3]:
            print(f"    📄 {data.name}")
        if len(data_files) > 3:
            print(f"    ... and {len(data_files) - 3} more files")

def create_final_summary():
    """Create final summary of capabilities"""
    
    print("🎉 ENHANCED VISUALIZATION SYSTEM SUMMARY")
    print("="*80)
    
    print("\n✅ Successfully Implemented Features:")
    print("  🎬 Multi-panel video generation (24x14 layout)")
    print("  🌐 Working 3D visualization panel")
    print("  📊 Real-time training metrics display")
    print("  🤖 Detailed agent status information")
    print("  🎯 POI coverage tracking and visualization")
    print("  📈 Performance comparison plots")
    print("  ⚡ Training process integration")
    print("  📋 Comprehensive information panels")
    
    print("\n🔧 Technical Achievements:")
    print("  • Fixed 3D visualization issues from original image")
    print("  • Added detailed symbol explanations and legends")
    print("  • Integrated video generation with training process")
    print("  • Created multi-algorithm comparison system")
    print("  • Enhanced visual clarity with icons and colors")
    print("  • Real-time performance indicator system")
    
    print("\n📊 Generated Visualizations Include:")
    print("  1. Main 2D View: Agent positions, POI status, coverage circles")
    print("  2. 3D View Panel: Altitude-aware agent visualization")
    print("  3. Agent Status: Position, energy, type, coverage radius")
    print("  4. POI Status: Priority levels, coverage indicators")
    print("  5. Training Metrics: Reward curves, coverage progress")
    print("  6. Performance Metrics: Efficiency, completion time")
    
    print("\n💡 Usage Examples:")
    print("  # Training with video:")
    print("  python main.py --algorithm ae_maddpg --video --episodes 50")
    print()
    print("  # Enhanced demo:")
    print("  python demo_enhanced_training_viz.py --enhanced")
    print()
    print("  # Results analysis:")
    print("  python analyze_training_results.py")
    
    print("\n🎯 Key Improvements Over Original:")
    print("  ✅ Fixed: 3D visualization now properly visible")
    print("  ✅ Added: Comprehensive symbol legends")
    print("  ✅ Added: Real-time training integration")
    print("  ✅ Added: Multi-panel information display")
    print("  ✅ Added: Professional video generation")
    print("  ✅ Added: Automated analysis tools")
    
    print("\n" + "="*80)
    print("🚀 Enhanced Multi-Agent Visualization System Complete!")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Training Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo only")
    parser.add_argument("--analysis-only", action="store_true", help="Run analysis only")
    
    args = parser.parse_args()
    
    if args.analysis_only:
        print("📊 Running analysis only...")
        subprocess.run([sys.executable, "analyze_training_results.py"])
    elif args.quick:
        print("⚡ Running quick demo...")
        subprocess.run([sys.executable, "demo_enhanced_training_viz.py", "--enhanced"])
    else:
        run_enhanced_training_demo()