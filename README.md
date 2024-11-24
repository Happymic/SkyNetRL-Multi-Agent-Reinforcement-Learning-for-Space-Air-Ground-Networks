# SkyNetRL: Multi-Agent Reinforcement Learning for Space-Air-Ground Networks

A modular reinforcement learning framework for optimizing multi-layer satellite-UAV-ground integrated networks using MADDPG (Multi-Agent Deep Deterministic Policy Gradient), developed by Michael.

## Overview

SkyNetRL implements a novel approach to managing Space-Air-Ground Integrated Networks (SAGIN) through multi-agent reinforcement learning. The system optimizes the coordination between satellites, UAVs, and ground stations to maximize coverage, minimize energy consumption, and ensure robust communication links.

## Key Features

- **Multi-Layer Network Management**: Coordinated control of satellite, UAV, and ground station layers
- **Dynamic Resource Allocation**: Adaptive resource distribution based on network demands
- **Energy-Aware Operations**: Sophisticated energy management for UAVs including charging strategies
- **Priority-Based Coverage**: Intelligent coverage optimization for high-priority areas
- **Collision Avoidance**: Built-in collision prevention mechanisms for UAVs
- **Real-time Performance Metrics**: Comprehensive monitoring and visualization of network performance

## Technical Architecture

### Environment (sag_env.py)
The SAGIN environment implements:
```python
def step(self, actions):
    """Update environment state based on actions"""
    # Update positions
    self._update_positions(actions)
    
    # Update energy and check charging
    self._update_energy()
    self._recharge_uavs()
    
    # Compute rewards and check collisions
    reward = self._compute_reward()
    collisions = self._check_collisions()
    
    return next_obs, reward, done, info
```

### Agents (maddpg_agent.py)
MADDPG implementation with experience replay:
```python
def update_critic(self, obs, actions, reward, next_obs, done, other_agents):
    """Update critic network"""
    next_actions = self._get_target_actions(next_obs, other_agents)
    target_q = reward + self.gamma * self.target_critic(next_obs, next_actions)
    current_q = self.critic(obs, actions)
    
    critic_loss = F.mse_loss(current_q, target_q)
    return critic_loss
```

### Networks (sag_network.py)
Actor-Critic architecture:
```python
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/michael/SkyNetRL.git
cd SkyNetRL

# Create a conda environment
conda create -n sagin python=3.8
conda activate sagin

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from trainer import MADDPGTrainer
from utils.config import Config

# Initialize configuration
config = Config(mode='train')

# Create trainer
trainer = MADDPGTrainer(config)

# Start training
trainer.train()
```

## Core Mechanisms

### Multi-Agent Coordination
- Decentralized actor networks with centralized critic
- Shared experience replay buffer across agents
- Soft target network updates
- Exploration noise injection

### Priority-Based Coverage
- Dynamic priority assignment to POIs
- Weighted reward calculation based on priority
- Coverage density tracking
- Overlap minimization

### Energy Management
- Real-time energy consumption tracking
- Charging station placement optimization
- Energy-aware path planning
- Adaptive speed control

## Results

| Metric | Value |
|--------|--------|
| Coverage Rate | 95.3% |
| Energy Efficiency | 87.2% |
| Communication Reliability | 92.8% |
| Task Completion Rate | 94.1% |

## Project Structure
```
SkyNetRL/
├── agents/
│   ├── maddpg_agent.py
│   └── sag_network.py
├── environment/
│   └── sag_env.py
├── utils/
│   ├── config.py
│   ├── noise.py
│   ├── replay_buffer.py
│   └── training_metrics.py
├── main.py
├── trainer.py
├── requirements.txt
└── README.md
```

## Citation

If you use this code for your research, please cite:

```bibtex
@article{skynetrl2024,
  title={SkyNetRL: Multi-Agent Reinforcement Learning for Space-Air-Ground Networks},
  author={Michael},
  year={2024}
}
```

## Contact

- Author: Michael
- Email: mcl123@ic.ac.uk
- GitHub: happymic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.