# agent.py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buf.append((torch.FloatTensor(s), a, r,
                         torch.FloatTensor(s2), done))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,s2,d = zip(*batch)
        return (torch.stack(s), torch.LongTensor(a),
                torch.FloatTensor(r), torch.stack(s2),
                torch.FloatTensor(d))
    def __len__(self):
        return len(self.buf)

class DQNAgent:
    def __init__(self, obs_dim, n_actions=4, arch='dqn',
                 lr=1e-2, gamma=0.99, batch_size=64,
                 buffer_size=5000, target_update=100,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        self.arch          = arch
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.n_actions     = n_actions
        self.steps_done    = 0
        self.online  = QNetwork(obs_dim, n_actions)
        self.target  = QNetwork(obs_dim, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_size)

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.n_actions-1)
        with torch.no_grad():
            return self.online(
                torch.FloatTensor(state).unsqueeze(0)).argmax().item()

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, float(done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s,a,r,s2,d = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            if self.arch == 'dqn':
                q_next = self.target(s2).max(1)[0]
            else:
                best_a = self.online(s2).argmax(1)
                q_next = self.target(s2).gather(
                    1, best_a.unsqueeze(1)).squeeze(1)
            td_target = r + self.gamma * q_next * (1.0 - d)
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = self.criterion(q_pred, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)