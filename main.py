from utils.config import Config
from trainer import MADDPGTrainer


def main():
    config = Config()
    trainer = MADDPGTrainer(config)

    # Train the model
    trainer.train()

    # Evaluate the final model
    trainer.load_models('final')
    eval_reward, eval_coverage = trainer.evaluate()
    print(f"Final Evaluation - Avg Reward: {eval_reward:.2f}, Avg Coverage: {eval_coverage:.2f}")


if __name__ == "__main__":
    main()