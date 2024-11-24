# agents/sag_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()

        # 打印网络维度信息以便调试
        print(f"Creating ActorNetwork with dims: obs={obs_dim}, action={action_dim}, hidden={hidden_dim}")

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 打印输入tensor的形状以便调试
        # print(f"Actor input shape: {x.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 使用tanh确保动作在[-1,1]范围内


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()

        print(f"Creating CriticNetwork with dims: obs={obs_dim}, action={action_dim}, hidden={hidden_dim}")

        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)