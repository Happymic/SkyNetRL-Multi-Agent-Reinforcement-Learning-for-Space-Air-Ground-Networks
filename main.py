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

def main():
    config = Config()

    # 设置随机种子以确保可重现性
    set_seed(config.seed)

    trainer = MADDPGTrainer(config)

    # 训练模型
    trainer.train()
    # Evaluate the final model
    trainer.load_models('best')
    eval_reward, eval_coverage = trainer.evaluate()
    print(f"Best Model Evaluation - Avg Reward: {eval_reward:.2f}, Avg Coverage: {eval_coverage:.2f}")

    # Visualize the best model
    trainer.visualizer.visualize(trainer.env, trainer.agents, 'best')


if __name__ == "__main__":
    main()