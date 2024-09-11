import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CommunicationModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        return torch.bmm(adj, self.fc(x))

class SAGNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.attention = MultiHeadAttention(hidden_dim, 4)  # 4 attention heads
        self.comm = CommunicationModule(hidden_dim)

    def forward(self, obs, adj):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # Apply attention mechanism
        x = self.attention(x, adj)

        # Apply communication
        x = self.comm(x, adj)

        return self.fc3(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        batch_size, num_agents, _ = x.shape

        q = self.query(x).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, num_agents, self.hidden_dim)
        return self.fc_out(out)