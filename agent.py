"""
agent.py
========
DQN and Double DQN agents with:
- 3-layer fully connected network [256, 128, 64]
- He initialisation
- Experience replay buffer (50,000)
- Epsilon-greedy exploration
- Hard target network update every 500 steps
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ── Neural Network ────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),     nn.ReLU(),
            nn.Linear(128, 64),      nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        # He initialisation
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch      = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, obs_dim, n_actions=4, arch='dqn',
                 lr=1e-3, gamma=0.99, batch_size=64,
                 buffer_size=50_000, target_update=500,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995):

        self.arch          = arch          # 'dqn' or 'ddqn'
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.n_actions     = n_actions
        self.steps_done    = 0

        self.online = QNetwork(obs_dim, n_actions)
        self.target = QNetwork(obs_dim, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=3e-3)
        self.criterion = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.online(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        """One gradient step. Returns loss value or None."""
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        # Compute TD target
        with torch.no_grad():
            if self.arch == 'dqn':
                # DQN: max Q from target network
                q_next = self.target(s2).max(1)[0]
            else:
                # DDQN: online selects action, target evaluates it
                best_actions = self.online(s2).argmax(1)
                q_next = self.target(s2).gather(
                    1, best_actions.unsqueeze(1)).squeeze(1)

            td_target = r + self.gamma * q_next * (1.0 - d)

        # Predicted Q for taken actions
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = self.criterion(q_pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

        # Hard target network update
        if self.steps_done % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
