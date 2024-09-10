import numpy as np
from tqdm import tqdm
import os
from agents.maddpg_agent import MADDPGAgent
from environment.sag_env import SAGEnvironment
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
from utils.visualizer import Visualizer

class MADDPGTrainer:
    def __init__(self, config):
        self.config = config
        self.env = SAGEnvironment(config)

        obs_dim = config.individual_obs_dim
        action_dim = config.action_dim

        self.agents = [MADDPGAgent(obs_dim, action_dim, config, i)
                       for i in range(config.num_agents)]
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.num_agents, obs_dim, action_dim)
        self.noise = OUNoise(action_dim, config.num_agents)
        self.visualizer = Visualizer(config)

    def train(self):
        pbar = tqdm(range(self.config.num_episodes), desc="Training")
        for episode in pbar:
            obs = self.env.reset()
            self.noise.reset()
            episode_reward = 0
            poi_coverage = []

            for step in range(self.config.max_time_steps):
                actions = np.array([agent.act(obs[i], noise_scale=self.noise.noise()[i]) for i, agent in enumerate(self.agents)])
                next_obs, rewards, done, info = self.env.step(actions)

                self.replay_buffer.add(obs, actions, rewards, next_obs, done)

                if len(self.replay_buffer) > self.config.batch_size:
                    sample = self.replay_buffer.sample(self.config.batch_size)
                    for i, agent in enumerate(self.agents):
                        other_agents = self.agents[:i] + self.agents[i+1:]
                        agent.update(sample, other_agents)
                        agent.update_targets()

                obs = next_obs
                episode_reward += np.sum(rewards)
                poi_coverage.append(info['coverage'])

                if done:
                    break

            if episode % self.config.log_frequency == 0:
                avg_coverage = np.mean(poi_coverage)
                pbar.set_postfix({
                    'Reward': f'{episode_reward:.2f}',
                    'PoI Coverage': f'{avg_coverage:.2f}',
                    'UAV Energy': f"{info['uav_energy']:.2f}"
                })

            if episode % self.config.eval_frequency == 0:
                eval_reward, eval_coverage = self.evaluate()
                pbar.write(f"Evaluation - Episode {episode}, Avg Reward: {eval_reward:.2f}, Avg Coverage: {eval_coverage:.2f}")

            if episode % self.config.visualize_frequency == 0:
                self.visualizer.visualize(self.env, self.agents, episode)

            if episode % self.config.save_frequency == 0:
                self.save_models(episode)

        self.save_models('final')

    def evaluate(self):
        total_reward = 0
        total_coverage = 0
        for _ in range(self.config.eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_coverage = []
            done = False
            while not done:
                actions = np.array([agent.act(obs[i], noise_scale=0) for i, agent in enumerate(self.agents)])
                obs, rewards, done, info = self.env.step(actions)
                episode_reward += np.sum(rewards)
                episode_coverage.append(info['coverage'])
            total_reward += episode_reward
            total_coverage += np.mean(episode_coverage)

        avg_reward = total_reward / self.config.eval_episodes
        avg_coverage = total_coverage / self.config.eval_episodes
        return avg_reward, avg_coverage

    def save_models(self, episode):
        save_path = os.path.join(self.config.model_save_path, f"episode_{episode}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(save_path, f"agent_{i}.pth"))

    def load_models(self, episode):
        load_path = os.path.join(self.config.model_save_path, f"episode_{episode}")
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(load_path, f"agent_{i}.pth"))