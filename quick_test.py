"""Quick test script to verify the training pipeline works correctly"""

from quick_test_config import QuickTestConfig
from minimal_trainer import MinimalMADDPGTrainer
import torch
import numpy as np
import random
import time


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_quick_test():
    print("="*60)
    print("Starting quick test of SkyNetRL training pipeline")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create minimal config
    config = QuickTestConfig()
    
    print("\nConfiguration Summary:")
    print(f"- Episodes: {config.num_episodes}")
    print(f"- Max steps per episode: {config.max_time_steps}")
    print(f"- Number of agents: {config.num_agents}")
    print(f"- Device: {config.device}")
    print(f"- Area size: {config.area_size}x{config.area_size}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    start_time = time.time()
    
    try:
        trainer = MinimalMADDPGTrainer(config)
        print("✓ Trainer initialized successfully")
        
        # Run training
        print("\nStarting minimal training...")
        trainer.train()
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "="*60)
        print("✓ QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nTotal time: {training_time:.2f} seconds")
        print(f"Results saved in: {config.base_dir}")
        
        # Print some basic metrics if available
        if hasattr(trainer, 'metrics'):
            summary = trainer.metrics.get_summary()
            print("\nFinal metrics summary:")
            for category in summary:
                if category == "Episode Metrics":
                    print(f"\n{category}:")
                    for metric, values in summary[category].items():
                        print(f"  {metric}: {values['current']:.3f}")
                    break
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_quick_test()
    exit(0 if success else 1)