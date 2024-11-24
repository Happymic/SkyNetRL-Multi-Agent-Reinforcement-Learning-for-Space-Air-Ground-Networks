import torch
import torch.nn.functional as F
import numpy as np
from agents.sag_network import ActorNetwork, CriticNetwork


class MADDPGAgent(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super(MADDPGAgent, self).__init__()

        # 打印初始化信息以便调试
        print(f"Initializing agent {agent_id} with dims: obs={obs_dim}, action={action_dim}")

        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.device = config.device

        # 创建网络
        self.actor = ActorNetwork(obs_dim, action_dim, config.hidden_dim)
        self.critic = CriticNetwork(obs_dim * config.num_agents,
                                    action_dim * config.num_agents,
                                    config.hidden_dim)
        self.target_actor = ActorNetwork(obs_dim, action_dim, config.hidden_dim)
        self.target_critic = CriticNetwork(obs_dim * config.num_agents,
                                           action_dim * config.num_agents,
                                           config.hidden_dim)

        # 移动网络到指定设备
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

        # 复制参数到目标网络
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

        # 创建优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=config.critic_lr)

        # 初始化噪声
        self.noise = config.exploration_noise

    def forward(self, obs):
        """前向传播"""
        return self.actor(obs)

    def act(self, obs, add_noise=True):
        """根据观察选择动作"""
        self.actor.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            action = self.actor(obs)
            if add_noise:
                noise = torch.randn_like(action) * self.noise
                action = torch.clamp(action + noise, -1, 1)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update_critic(self, obs, actions, reward, next_obs, done, other_agents):
        """更新评论家网络"""
        # 准备目标Q值
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate([self] + other_agents):
                next_obs_i = next_obs[:, i]
                next_action_i = agent.target_actor(next_obs_i)
                next_actions.append(next_action_i)
            next_actions = torch.cat(next_actions, dim=1)

            next_state = next_obs.reshape(next_obs.size(0), -1)
            next_actions = next_actions.reshape(next_actions.size(0), -1)

            target_q = reward.unsqueeze(1) + (1 - done.float()) * \
                       self.config.gamma * self.target_critic(next_state, next_actions)

        # 计算当前Q值
        current_state = obs.reshape(obs.size(0), -1)
        current_actions = actions.reshape(actions.size(0), -1)
        current_q = self.critic(current_state, current_actions)

        # 计算critic损失
        critic_loss = F.mse_loss(current_q, target_q.detach())

        # 更新critic
        self.critic_optimizer.zero_grad()
        if self.config.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                critic_loss.backward()
        else:
            critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, obs, actions, agent_index, other_agents):
        """更新演员网络"""
        current_actions = actions.clone()
        current_actions[:, agent_index] = self.actor(obs[:, agent_index])

        state = obs.reshape(obs.size(0), -1)
        actions = current_actions.reshape(current_actions.size(0), -1)

        actor_loss = -self.critic(state, actions).mean()

        # 更新actor
        self.actor_optimizer.zero_grad()
        if self.config.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                actor_loss.backward()
        else:
            actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_targets(self):
        """软更新目标网络"""
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target, source):
        """执行软更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) +
                param.data * self.config.tau
            )

    def decay_noise(self):
        """衰减探索噪声"""
        self.noise *= self.config.exploration_decay

    def _hard_update(self, target, source):
        """硬更新：直接复制参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])