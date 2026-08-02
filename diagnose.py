"""
diagnose.py
===========
Quick diagnostic: does the baseline DQN agent learn AT ALL if given
more episodes? Runs ONE seed, prints rolling success rate every 50
episodes, so we can see if it's climbing (needs more time) or stuck
flat at 0% (something's broken).

Takes ~10-20 minutes on Colab GPU, much less than a full 8-config run.

Save this file in the SAME folder as env.py and agent.py
(i.e. the root of your repo, next to train.py / run_all.py).

Run:
    python diagnose.py
"""

import random
import numpy as np
from collections import deque

from env   import GridWorld
from agent import DQNAgent

SEED        = 1
N_EPISODES  = 2000     # much longer than the usual 500, just to see the trend
PRINT_EVERY = 50

random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)

env   = GridWorld(grid_size=15, obstacle_density=0.20,
                   slip_prob=0.10, max_steps=500, seed=SEED)
agent = DQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions,
                  arch='dqn', gamma=0.99)

success_window = deque(maxlen=50)
outcome_counts = {'goal': 0, 'collision': 0, 'timeout': 0}

print("=" * 60)
print(f"Diagnostic run: DQN baseline, seed={SEED}, {N_EPISODES} episodes")
print("=" * 60)

for ep in range(1, N_EPISODES + 1):
    state = env.reset()
    done  = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, float(done))
        agent.update()
        state = next_state

    agent.decay_epsilon()

    success = 1 if info == 'goal' else 0
    success_window.append(success)
    outcome_counts[info] = outcome_counts.get(info, 0) + 1

    if ep % PRINT_EVERY == 0:
        rolling = np.mean(success_window)
        print(f"  ep {ep:4d} | rolling success (last 50) = {rolling:.2f} "
              f"| eps={agent.eps:.3f} "
              f"| goals={outcome_counts['goal']:4d} "
              f"collisions={outcome_counts['collision']:4d} "
              f"timeouts={outcome_counts['timeout']:4d}")

print("\n" + "=" * 60)
print("DONE. Read the trend above:")
print("  - If rolling success climbs over time  -> agent IS learning,")
print("    just needs more episodes than 500.")
print("  - If it stays flat near 0.00 the whole run -> something else")
print("    is wrong (reward signal, obs encoding, hyperparams, etc).")
print("=" * 60)
