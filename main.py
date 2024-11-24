from utils.config import Config
from trainer import MADDPGTrainer
import torch
import numpy as np
import random
import os

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
    config = Config()
    trainer = MADDPGTrainer(config)

    try:
        print("Starting training...")
        trainer.train()

        # 训练结束后生成完整报告
        final_summary = trainer.metrics.get_summary()
        print("\n=== Final Training Results ===")
        for category in final_summary:
            print(f"\n{category}:")
            for metric, values in final_summary[category].items():
                print(f"  {metric}:")
                print(f"    Final Value: {values['current']:.3f}")
                print(f"    Mean: {values['mean']:.3f} ± {values['std']:.3f}")
                print(f"    Best: {values['max']:.3f}")

        # 生成最终的可视化报告
        trainer.metrics.plot_metrics(
            save_path=os.path.join(config.base_dir, "final_training_metrics.html")
        )

        print("\nTraining completed successfully!")
        print(f"Metrics and visualizations saved to: {config.base_dir}")

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()