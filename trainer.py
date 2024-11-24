import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import os
import torch
from agents.maddpg_agent import MADDPGAgent
from environment.sag_env import SAGEnvironment
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
from utils.visualizer import Visualizer
from utils.training_metrics import TrainingMetrics

class MADDPGTrainer:
    def __init__(self, config):
        """初始化MADDPG训练器"""
        self.config = config
        self.device = config.device

        # 初始化环境
        self.env = SAGEnvironment(config)

        # 设置智能体参数
        obs_dim = config.individual_obs_dim
        action_dim = config.action_dim

        # 初始化智能体
        self.agents = [
            MADDPGAgent(obs_dim, action_dim, config, i).to(self.device)
            for i in range(config.num_agents)
        ]

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            config.num_agents,
            obs_dim,
            action_dim
        )

        # 初始化探索噪声
        self.noise = OUNoise(
            action_dim,
            config.num_agents,
            scale=config.exploration_noise
        )

        # 初始化可视化器和指标收集器
        self.visualizer = Visualizer(config)
        self.metrics = TrainingMetrics(config)

        # 初始化最佳奖励记录
        self.best_reward = float('-inf')
        self.best_episode = 0

        # 创建保存目录
        self.save_dir = os.path.join(config.base_dir, "training_data")
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        """训练主循环"""
        print("\nStarting training with configuration:")
        print(f"Number of agents: {self.config.num_agents}")
        print(f"Device: {self.device}")
        print(f"Number of episodes: {self.config.num_episodes}")
        print(f"Max timesteps per episode: {self.config.max_time_steps}")

        pbar = tqdm(range(self.config.num_episodes), desc="Training")

        for episode in pbar:
            # 重置环境和噪声
            obs = self.env.reset()
            self.noise.reset()
            episode_reward = 0

            # 收集episode数据
            episode_data = {
                'coverage': [],
                'energy': [],
                'positions': [],
                'collisions': 0,
                'rewards': []
            }

            # Episode循环
            for step in range(self.config.max_time_steps):
                # 将观察转换为tensor并移动到正确设备
                obs_tensor = torch.FloatTensor(obs).to(self.device)

                # 获取动作
                actions = []
                for i, agent in enumerate(self.agents):
                    if self.config.use_mixed_precision:
                        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                            action = agent.act(obs_tensor[i], add_noise=True)
                    else:
                        action = agent.act(obs_tensor[i], add_noise=True)
                    actions.append(action)
                actions = np.array(actions)

                # 环境交互
                next_obs, rewards, done, info = self.env.step(actions)

                # 收集数据
                episode_data['coverage'].append(info['coverage'])
                episode_data['energy'].append(info['energy'])
                episode_data['positions'] = info['positions']
                episode_data['collisions'] = info['collisions']
                episode_data['rewards'].append(rewards)

                # 存储经验
                self.replay_buffer.add(obs, actions, rewards, next_obs, done)

                # 如果经验池足够大，进行学习
                if len(self.replay_buffer) > self.config.batch_size:
                    if step % self.config.gradient_accumulation_steps == 0:
                        self._update_agents()

                obs = next_obs
                episode_reward += np.sum(rewards)

                # 更新可视化数据
                self.visualizer.update_env_data(self.env, episode, step)

                if done:
                    break

            # 更新指标
            self.metrics.update(self.env, episode_data)

            # 衰减探索噪声
            for agent in self.agents:
                agent.decay_noise()

            # 更新进度条信息
            avg_coverage = np.mean(episode_data['coverage'])
            avg_energy = np.mean(episode_data['energy'])
            pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Coverage': f'{avg_coverage:.2f}',
                'Energy': f'{avg_energy:.2f}',
                'Collisions': episode_data['collisions']
            })

            # 定期评估
            if episode % self.config.eval_frequency == 0:
                eval_reward, eval_metrics = self.evaluate()
                self._log_evaluation(episode, eval_reward, eval_metrics)

                # 更新最佳模型
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.best_episode = episode
                    self.save_models('best')

            # 定期保存
            if episode % self.config.save_frequency == 0:
                self.save_models(f'episode_{episode}')
                self.save_episode_data(episode, episode_data)

            # 定期清理GPU内存
            if episode % self.config.empty_cache_freq == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 训练结束
        self.save_models('final')
        self.metrics.save_metrics()
        self._generate_final_report()

        return self.best_reward, self.best_episode

    def _update_agents(self):
        """更新所有智能体的网络"""
        sample = self.replay_buffer.sample(self.config.batch_size)

        # 将sample移动到正确的设备
        obs, actions, rewards, next_obs, dones = [
            torch.FloatTensor(x).to(self.device) for x in sample
        ]

        for i, agent in enumerate(self.agents):
            # 获取其他智能体列表
            other_agents = self.agents[:i] + self.agents[i + 1:]

            if self.config.use_mixed_precision:
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    # 更新critic
                    critic_loss = agent.update_critic(
                        obs, actions, rewards[:, i], next_obs, dones,
                        other_agents
                    )

                    # 更新actor
                    actor_loss = agent.update_actor(
                        obs, actions, i, other_agents
                    )

                    # 软更新目标网络
                    agent.update_targets()
            else:
                critic_loss = agent.update_critic(
                    obs, actions, rewards[:, i], next_obs, dones,
                    other_agents
                )
                actor_loss = agent.update_actor(
                    obs, actions, i, other_agents
                )
                agent.update_targets()


    def evaluate(self, num_episodes=None):
        """评估当前策略"""
        if num_episodes is None:
            num_episodes = self.config.eval_episodes

        total_reward = 0
        eval_metrics = {
            'coverage': [],
            'energy': [],
            'collisions': 0
        }

        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # 获取动作（无探索）
                actions = []
                for i, agent in enumerate(self.agents):
                    obs_tensor = torch.FloatTensor(obs[i]).to(self.device)
                    with torch.no_grad():
                        action = agent.act(obs_tensor, add_noise=False)
                    actions.append(action)
                actions = np.array(actions)

                # 执行动作
                obs, rewards, done, info = self.env.step(actions)
                episode_reward += np.sum(rewards)

                # 收集评估指标
                eval_metrics['coverage'].append(info['coverage'])
                eval_metrics['energy'].append(info['energy'])
                eval_metrics['collisions'] += info['collisions']

            total_reward += episode_reward

        return total_reward / num_episodes, eval_metrics

    def save_models(self, tag):
        """保存模型"""
        save_path = os.path.join(self.config.model_save_path, f"checkpoint_{tag}")
        os.makedirs(save_path, exist_ok=True)

        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(save_path, f"agent_{i}.pth")
            agent.save(agent_path)

    def load_models(self, tag):
        """加载模型"""
        load_path = os.path.join(self.config.model_save_path, f"checkpoint_{tag}")

        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(load_path, f"agent_{i}.pth")
            agent.load(agent_path)

    def save_episode_data(self, episode, episode_data):
        """保存episode数据"""
        data = {
            "episode": episode,
            "coverage": np.mean(episode_data['coverage']),
            "energy": np.mean(episode_data['energy']),
            "collisions": episode_data['collisions'],
            "rewards": np.mean(episode_data['rewards']),
            "timestamp": datetime.now().isoformat()
        }

        filename = os.path.join(self.save_dir, f"episode_{episode}.json")
        with open(filename, 'w') as f:
            json.dump(data, f)

    def _log_evaluation(self, episode, eval_reward, eval_metrics):
        """记录评估结果"""
        print(f"\nEvaluation - Episode {episode}")
        print(f"Average Reward: {eval_reward:.2f}")
        print(f"Average Coverage: {np.mean(eval_metrics['coverage']):.2f}")
        print(f"Average Energy: {np.mean(eval_metrics['energy']):.2f}")
        print(f"Total Collisions: {eval_metrics['collisions']}")

    def _generate_final_report(self):
        """生成训练最终报告"""
        print("\n=== Training Complete ===")
        print(f"Best Episode: {self.best_episode}")
        print(f"Best Reward: {self.best_reward:.2f}")

        # 获取最终指标摘要
        final_summary = self.metrics.get_summary()
        print("\nFinal Metrics Summary:")
        for category, metrics in final_summary.items():
            print(f"\n{category}:")
            for metric_name, values in metrics.items():
                print(f"  {metric_name}:")
                print(f"    Final: {values['current']:.3f}")
                print(f"    Mean: {values['mean']:.3f} ± {values['std']:.3f}")
                print(f"    Range: [{values['min']:.3f}, {values['max']:.3f}]")

        # 保存最终报告
        report_path = os.path.join(self.config.base_dir, "final_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'best_episode': self.best_episode,
                'best_reward': float(self.best_reward),
                'final_metrics': final_summary
            }, f, indent=4)

        print(f"\nFinal report saved to: {report_path}")

    def print_memory_stats(self):
        """打印内存使用状况"""
        if torch.cuda.is_available():
            print("\nGPU Memory Stats:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB")