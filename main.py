from utils.config import Config
from trainer import MADDPGTrainer
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set CUDA deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Check and return the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Print GPU information
        gpu_properties = torch.cuda.get_device_properties(device)
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_properties.total_memory / 1024 ** 3:.2f} GB")
        print(f"GPU Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}\n")
    else:
        device = torch.device("cpu")
        print("\nNo GPU available, using CPU instead.\n")
    return device


def main():
    # Get the best available device
    device = get_device()

    # Create config
    config = Config()

    # Add device to config for use throughout the project
    config.device = device

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Optimize CUDA settings for training
    if device.type == 'cuda':
        # Enable TF32 precision for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set optimal memory allocation strategy
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        # Print CUDA memory status before training
        print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Initial CUDA memory cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB\n")

    # Create trainer with device configuration
    trainer = MADDPGTrainer(config)

    # Train model
    print("Starting training...")
    trainer.train()

    # Evaluate the final model
    print("\nEvaluating best model...")
    trainer.load_models('best')
    eval_reward, eval_coverage = trainer.evaluate()
    print(f"Best Model Evaluation - Avg Reward: {eval_reward:.2f}, Avg Coverage: {eval_coverage:.2f}")

    # If using GPU, print final memory status
    if device.type == 'cuda':
        print(f"\nFinal CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Final CUDA memory cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

    # Visualize the best model
    print("\nStarting visualization...")
    trainer.visualizer.visualize(trainer.env, trainer.agents, 'best')


if __name__ == "__main__":
    main()