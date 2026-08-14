"""
train.py
========
Single training run. Built on same logic as test_basic.py.
"""
import random
import numpy as np
from collections import deque

from env     import GridWorld
from agent   import DQNAgent
from shaping import SHAPING_MAP, get_shaped_reward


def run_one(arch='dqn', shaping='baseline', seed=1,
            n_episodes=600, gamma=0.99):

    random.seed(seed)
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    env   = GridWorld(seed=seed)
    agent = DQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions,
                     arch=arch, gamma=gamma)

    phi_fn = SHAPING_MAP[shaping]

    success_window     = deque(maxlen=50)
    lc                 = []
    ep_losses          = []
    last200_losses     = []
    last200_successes  = []
    last200_collisions = []
    ce                 = n_episodes

    for ep in range(n_episodes):
        state    = env.reset()
        done     = False
        t        = 0
        ep_loss  = []

        phi_prev = phi_fn(env, t) if phi_fn is not None else 0.0

        while not done:
            action              = agent.select_action(state)
            next_state, r_base, done, info = env.step(action)
            t += 1

            # Shaped reward
            if phi_fn is not None:
                phi_next = phi_fn(env, t)
                r_shaped = r_base + gamma * phi_next - phi_prev
                phi_prev = phi_next
            else:
                r_shaped = r_base

            agent.store(state, action, r_shaped, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)

            state = next_state

        agent.decay_epsilon()

        success   = 1 if info == 'goal'      else 0
        collision = 1 if info == 'collision' else 0

        success_window.append(success)
        rolling = np.mean(success_window)
        lc.append(rolling)
        ep_losses.append(np.mean(ep_loss) if ep_loss else 0.0)

        if ce == n_episodes and len(success_window) == 50 and rolling >= 0.60:
            ce = ep

        if ep >= n_episodes - 200:
            if ep_loss:
                last200_losses.extend(ep_loss)
            last200_successes.append(success)
            last200_collisions.append(collision)

    final_mse = float(np.mean(last200_losses)) if last200_losses else float('nan')
    n_suc     = sum(last200_successes)
    n_col     = sum(last200_collisions)
    scr       = n_suc / n_col if n_col > 0 else float(n_suc)

    return ce, np.array(lc), np.array(ep_losses), final_mse, scr


if __name__ == '__main__':
    print("Testing DQN baseline, seed=1 ...")
    ce, lc, losses, mse, scr = run_one('dqn', 'baseline', seed=1)
    print(f"CE={ce}  MSE={mse:.4f}  SCR={scr:.2f}")
    print("Last 5 rolling success rates:", lc[-5:].round(2))

    next_state, r_base, done, info = env.step(action)
    print(f"r_base={r_base}")  # ADD THIS
    t += 1