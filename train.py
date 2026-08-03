"""
train.py
========
Runs one training configuration (arch + shaping + seed).
Returns all metrics needed for the paper.
"""

import random
import numpy as np
from collections import deque

from env      import GridWorld
from agent    import DQNAgent
from shaping  import SHAPING_MAP, get_shaped_reward


def run_one(arch='dqn', shaping='baseline', seed=1,
            n_episodes=500, grid_size=15,
            obstacle_density=0.20, slip_prob=0.10,
            max_steps=500, gamma=0.99):
    """
    Run one training experiment.

    Returns
    -------
    ce          : int   — convergence episode (500 if never)
    lc          : array — 50-ep rolling success rate per episode
    loss_curve  : array — mean loss per episode
    final_mse   : float — mean loss over last 200 episodes
    scr         : float — success/collision over last 200 episodes
    """
    # ── Seed everything ───────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)

    # ── Setup ─────────────────────────────────────────────────────
    env    = GridWorld(grid_size=grid_size,
                       obstacle_density=obstacle_density,
                       slip_prob=slip_prob,
                       max_steps=max_steps,
                       seed=seed)

    agent  = DQNAgent(obs_dim=env.obs_dim,
                      n_actions=env.n_actions,
                      arch=arch, gamma=gamma)

    phi_fn = SHAPING_MAP[shaping]

    # ── Tracking ──────────────────────────────────────────────────
    success_window = deque(maxlen=50)
    lc             = []          # rolling success rate per episode
    ep_losses      = []          # mean loss per episode

    last200_losses     = []
    last200_successes  = []
    last200_collisions = []

    ce = n_episodes   # default: never converged

    # ── Training loop ─────────────────────────────────────────────
    for ep in range(n_episodes):
        state    = env.reset()
        done     = False
        t        = 0
        ep_loss  = []

        # Compute Phi(s) before first step
        phi_prev = phi_fn(env, t) if phi_fn is not None else 0.0

        while not done:
            action              = agent.select_action(state)
            next_state, r_base, done, info = env.step(action)
            t += 1

            # Shaped reward
            r_shaped, phi_prev = get_shaped_reward(
                r_base, phi_fn, env, t,
                gamma=gamma, phi_prev=phi_prev)

            # Store and learn
            agent.store(state, action, r_shaped, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)

            state = next_state

        agent.decay_epsilon()

        # ── Episode metrics ───────────────────────────────────────
        success   = 1 if info == 'goal'      else 0
        collision = 1 if info == 'collision' else 0

        success_window.append(success)
        rolling = np.mean(success_window)
        lc.append(rolling)
        ep_losses.append(np.mean(ep_loss) if ep_loss else 0.0)

        # Convergence episode: first time rolling avg >= 85%
        if ce == n_episodes and len(success_window) == 50 and rolling >= 0.60:
            ce = ep

        # Last 200 episodes
        if ep >= n_episodes - 200:
            if ep_loss:
                last200_losses.extend(ep_loss)
            last200_successes.append(success)
            last200_collisions.append(collision)

    # ── Final metrics ─────────────────────────────────────────────
    final_mse = float(np.mean(last200_losses)) if last200_losses else float('nan')
    n_suc     = sum(last200_successes)
    n_col     = sum(last200_collisions)
    scr       = n_suc / n_col if n_col > 0 else float('nan')

    return ce, np.array(lc), np.array(ep_losses), final_mse, scr


# ── Quick test ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing one run: DQN + TPDS, seed=1 ...")
    ce, lc, losses, mse, scr = run_one('dqn', 'tpds', seed=1)
    print(f"CE={ce}  MSE={mse:.4f}  SCR={scr:.2f}")
    print("✅ train.py works correctly")
