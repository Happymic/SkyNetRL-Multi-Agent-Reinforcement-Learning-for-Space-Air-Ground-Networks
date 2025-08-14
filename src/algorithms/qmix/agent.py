"""
QMIX Agent Implementation
Value decomposition approach for cooperative multi-agent reinforcement learning
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

from .networks import QMixAgent, QMixMixer, DoubleQMixMixer, QMixRNN


class QMIXMultiAgent:
    """QMIX Multi-Agent System with value function factorization"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, 
                 state_dim: int, config: Dict):
        """
        Initialize QMIX multi-agent system
        
        Args:
            num_agents: Number of agents
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent (discrete actions)
            state_dim: Global state dimension
            config: Configuration dictionary
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Discretize continuous actions
        self.num_discrete_actions = config.get('num_discrete_actions', 5)
        self.action_bounds = config.get('action_bounds', [-1, 1])
        
        # Create discrete action space
        self.discrete_actions = self._create_discrete_actions()
        self.total_discrete_actions = len(self.discrete_actions)
        
        # Create agent networks
        self.use_rnn = config.get('use_rnn', False)
        self.agents = []
        
        for i in range(num_agents):
            if self.use_rnn:
                agent = QMixRNN(obs_dim, self.total_discrete_actions, config).to(self.device)
            else:
                agent = QMixAgent(obs_dim, self.total_discrete_actions, config).to(self.device)
            self.agents.append(agent)
        
        # Create target networks
        self.target_agents = []
        for agent in self.agents:
            target_agent = copy.deepcopy(agent).to(self.device)
            # Freeze target network
            for param in target_agent.parameters():
                param.requires_grad = False
            self.target_agents.append(target_agent)
        
        # Create mixer networks
        mixer_type = config.get('mixer_type', 'standard')
        if mixer_type == 'double':
            self.mixer = DoubleQMixMixer(num_agents, state_dim, config).to(self.device)
            self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        else:
            self.mixer = QMixMixer(num_agents, state_dim, config).to(self.device)
            self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        
        # Freeze target mixer
        for param in self.target_mixer.parameters():
            param.requires_grad = False
        
        # Optimizers
        all_parameters = []
        for agent in self.agents:
            all_parameters.extend(list(agent.parameters()))
        all_parameters.extend(list(self.mixer.parameters()))
        
        self.optimizer = torch.optim.Adam(
            all_parameters,
            lr=config.get('lr', 3e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.grad_clip = config.get('grad_clip', 10.0)
        
        # RNN hidden states
        if self.use_rnn:
            self.hidden_states = [agent.init_hidden(1).to(self.device) for agent in self.agents]
            self.target_hidden_states = [agent.init_hidden(1).to(self.device) for agent in self.target_agents]
        
        # Training statistics
        self.training_step = 0
        self.training_stats = {
            'q_loss': [],
            'mixer_loss': [],
            'total_loss': [],
            'epsilon': []
        }
    
    def _create_discrete_actions(self) -> torch.Tensor:
        """Create discrete action space from continuous bounds"""
        actions = []
        # Create grid of discrete actions for 2D continuous space
        x_actions = torch.linspace(self.action_bounds[0], self.action_bounds[1], self.num_discrete_actions)
        y_actions = torch.linspace(self.action_bounds[0], self.action_bounds[1], self.num_discrete_actions)
        
        for x in x_actions:
            for y in y_actions:
                actions.append([x.item(), y.item()])
        
        return torch.tensor(actions, dtype=torch.float32)
    
    def act(self, observations: np.ndarray, add_noise: bool = True) -> Dict[int, np.ndarray]:
        """
        Generate actions for all agents
        
        Args:
            observations: [num_agents, obs_dim] - agent observations
            add_noise: Whether to add exploration noise (epsilon-greedy)
        
        Returns:
            actions: Dictionary of agent actions
        """
        actions = {}
        
        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                
                if self.use_rnn:
                    q_values, self.hidden_states[i] = agent(obs.unsqueeze(1), self.hidden_states[i])
                    q_values = q_values.squeeze(1)
                else:
                    q_values = agent(obs)
                
                # Epsilon-greedy action selection
                if add_noise and np.random.random() < self.epsilon:
                    action_idx = np.random.randint(0, self.total_discrete_actions)
                else:
                    action_idx = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                    action_idx = min(action_idx, self.total_discrete_actions - 1)  # Ensure bounds
                
                # Convert discrete action to continuous
                continuous_action = self.discrete_actions[action_idx].numpy()
                actions[i] = continuous_action
        
        return actions
    
    def update(self, batch: Dict) -> Dict:
        """
        Update QMIX networks
        
        Args:
            batch: Training batch with transitions
        
        Returns:
            training_info: Dictionary of training statistics
        """
        self.training_step += 1
        
        # Extract batch data
        states = batch['states'].to(self.device)  # [batch, num_agents, obs_dim]
        actions = batch['actions'].to(self.device)  # [batch, num_agents, action_dim]
        rewards = batch['rewards'].to(self.device)  # [batch, num_agents]
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)  # [batch, num_agents]
        global_states = batch.get('global_states', states.mean(dim=1)).to(self.device)  # [batch, state_dim]
        next_global_states = batch.get('next_global_states', next_states.mean(dim=1)).to(self.device)
        
        batch_size = states.shape[0]
        
        # Convert continuous actions to discrete action indices
        action_indices = self._actions_to_indices(actions)
        
        # Get current Q-values
        current_q_values = []
        for i, agent in enumerate(self.agents):
            if self.use_rnn:
                q_vals, _ = agent(states[:, i:i+1, :], None)  # Process as sequence
                q_vals = q_vals.squeeze(1)
            else:
                q_vals = agent(states[:, i, :])
            
            # Get Q-value for chosen action
            agent_q = q_vals.gather(1, action_indices[:, i:i+1])
            current_q_values.append(agent_q)
        
        current_q_values = torch.cat(current_q_values, dim=1)  # [batch, num_agents]
        
        # Get next Q-values for target
        with torch.no_grad():
            next_q_values = []
            for i, target_agent in enumerate(self.target_agents):
                if self.use_rnn:
                    next_q_vals, _ = target_agent(next_states[:, i:i+1, :], None)
                    next_q_vals = next_q_vals.squeeze(1)
                else:
                    next_q_vals = target_agent(next_states[:, i, :])
                
                # Use max Q-value for target
                max_next_q = torch.max(next_q_vals, dim=1, keepdim=True)[0]
                next_q_values.append(max_next_q)
            
            next_q_values = torch.cat(next_q_values, dim=1)  # [batch, num_agents]
        
        # Mix current Q-values
        qtot_current = self.mixer(current_q_values, global_states)
        
        # Mix target Q-values
        with torch.no_grad():
            qtot_next = self.target_mixer(next_q_values, next_global_states)
            # Use team reward (sum of individual rewards)
            team_rewards = rewards.sum(dim=1, keepdim=True)
            team_dones = dones.max(dim=1, keepdim=True)[0]  # Episode done if any agent is done
            targets = team_rewards + self.gamma * qtot_next * (1 - team_dones)
        
        # Compute loss
        td_loss = F.mse_loss(qtot_current, targets)
        
        # Add regularization
        reg_loss = 0.0
        for agent in self.agents:
            for param in agent.parameters():
                reg_loss += torch.norm(param) ** 2
        
        total_loss = td_loss + self.config.get('reg_weight', 1e-5) * reg_loss
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        all_parameters = []
        for agent in self.agents:
            all_parameters.extend(list(agent.parameters()))
        all_parameters.extend(list(self.mixer.parameters()))
        torch.nn.utils.clip_grad_norm_(all_parameters, self.grad_clip)
        
        self.optimizer.step()
        
        # Update target networks
        self._update_targets()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Record statistics
        self.training_stats['q_loss'].append(td_loss.item())
        self.training_stats['total_loss'].append(total_loss.item())
        self.training_stats['epsilon'].append(self.epsilon)
        
        training_info = {
            'q_loss': td_loss.item(),
            'total_loss': total_loss.item(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        
        return training_info
    
    def _actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert continuous actions to discrete action indices"""
        batch_size, num_agents = actions.shape[:2]
        action_indices = torch.zeros(batch_size, num_agents, dtype=torch.long, device=self.device)
        
        for b in range(batch_size):
            for a in range(num_agents):
                # Find closest discrete action
                distances = torch.norm(self.discrete_actions.to(self.device) - actions[b, a], dim=1)
                action_idx = torch.argmin(distances).item()
                # Ensure index is within bounds
                action_indices[b, a] = min(action_idx, self.total_discrete_actions - 1)
        
        return action_indices
    
    def _update_targets(self):
        """Soft update target networks"""
        # Update target agents
        for agent, target_agent in zip(self.agents, self.target_agents):
            for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update target mixer
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def reset_hidden_states(self):
        """Reset RNN hidden states (for episodic training)"""
        if self.use_rnn:
            for i in range(self.num_agents):
                self.hidden_states[i] = self.agents[i].init_hidden(1).to(self.device)
                self.target_hidden_states[i] = self.target_agents[i].init_hidden(1).to(self.device)
    
    def save(self, filepath: str):
        """Save QMIX networks and statistics"""
        state_dict = {
            'agents': [agent.state_dict() for agent in self.agents],
            'mixer': self.mixer.state_dict(),
            'target_agents': [agent.state_dict() for agent in self.target_agents],
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'config': self.config
        }
        torch.save(state_dict, filepath)
    
    def load(self, filepath: str):
        """Load QMIX networks and statistics"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(checkpoint['agents'][i])
        
        for i, target_agent in enumerate(self.target_agents):
            target_agent.load_state_dict(checkpoint['target_agents'][i])
        
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.training_stats = checkpoint['training_stats']
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
    
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        for agent in self.agents:
            agent.eval()
        self.mixer.eval()
    
    def set_train_mode(self):
        """Set networks to training mode"""
        for agent in self.agents:
            agent.train()
        self.mixer.train()
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics for analysis"""
        if not self.training_stats['q_loss']:
            return {}
        
        return {
            'avg_q_loss': np.mean(self.training_stats['q_loss'][-100:]),
            'avg_total_loss': np.mean(self.training_stats['total_loss'][-100:]),
            'current_epsilon': self.epsilon,
            'total_training_steps': self.training_step
        }