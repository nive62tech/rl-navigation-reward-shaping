"""
test_basic.py
=============
Absolute minimum test. No shaping, no classes, no imports.
Just: can a DQN learn to navigate a tiny grid?
Run: python test_basic.py
Should show goals increasing over time.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Tiny grid ────────────────────────────────────────────────────
N          = 5       # 5x5 grid — very easy
MAX_STEPS  = 50
N_EPISODES = 500
GOAL       = (N-1, N-1)

def reset():
    return [0, 0]

def observe(agent):
    gr, gc = GOAL
    r, c   = agent
    return torch.FloatTensor([
        r/N, c/N, gr/N, gc/N,
        (gr-r)/N, (gc-c)/N
    ])

def step(agent, action):
    r, c = agent
    if   action == 0: r = max(0,   r-1)   # up
    elif action == 1: r = min(N-1, r+1)   # down
    elif action == 2: c = max(0,   c-1)   # left
    elif action == 3: c = min(N-1, c+1)   # right
    agent[0], agent[1] = r, c
    if (r,c) == GOAL:
        return True, +1.0, 'goal'
    return False, 0.0, 'step'

# ── Network ───────────────────────────────────────────────────────
net    = nn.Sequential(nn.Linear(6,64), nn.ReLU(),
                       nn.Linear(64,64), nn.ReLU(),
                       nn.Linear(64,4))
target = nn.Sequential(nn.Linear(6,64), nn.ReLU(),
                       nn.Linear(64,64), nn.ReLU(),
                       nn.Linear(64,4))
target.load_state_dict(net.state_dict())

opt      = optim.Adam(net.parameters(), lr=1e-2)
loss_fn  = nn.MSELoss()
buf      = deque(maxlen=5000)
eps      = 1.0
goals    = 0

print("ep   | goals_so_far | eps  | last_50_success_rate")
print("-" * 52)

window = deque(maxlen=50)

for ep in range(1, N_EPISODES+1):
    agent = reset()
    obs   = observe(agent)
    done  = False
    steps = 0

    while not done and steps < MAX_STEPS:
        # Action
        if random.random() < eps:
            a = random.randint(0,3)
        else:
            with torch.no_grad():
                a = net(obs).argmax().item()

        done, r, info = step(agent, a)
        obs2 = observe(agent)
        buf.append((obs, a, r, obs2, float(done)))
        obs  = obs2
        steps += 1

        # Train
        if len(buf) >= 64:
            batch = random.sample(buf, 64)
            s_b, a_b, r_b, s2_b, d_b = zip(*batch)
            s_t  = torch.stack(s_b)
            a_t  = torch.LongTensor(a_b)
            r_t  = torch.FloatTensor(r_b)
            s2_t = torch.stack(s2_b)
            d_t  = torch.FloatTensor(d_b)

            with torch.no_grad():
                q_next = target(s2_t).max(1)[0]
                tgt    = r_t + 0.99 * q_next * (1-d_t)

            q_pred = net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
            loss   = loss_fn(q_pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    if info == 'goal':
        goals += 1

    window.append(1 if info=='goal' else 0)
    eps = max(0.05, eps * 0.995)

    # Hard update every 100 eps
    if ep % 100 == 0:
        target.load_state_dict(net.state_dict())

    if ep % 50 == 0:
        print(f"{ep:4d} | {goals:12d} | {eps:.3f} | {np.mean(window):.2f}")

print()
if goals > 50:
    print("✅ DQN IS learning. Problem was in env.py or shaping.py")
elif goals > 10:
    print("⚠️  DQN learning slowly. May need more episodes.")
else:
    print("❌ DQN not learning even on 5x5. PyTorch or environment issue.")