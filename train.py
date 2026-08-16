# train.py
import random
import numpy as np
from collections import deque
from env     import GridWorld
from agent   import DQNAgent
from shaping import SHAPING_MAP

def run_one(arch='dqn', shaping='baseline', seed=1,
            n_episodes=600, gamma=0.99):
    random.seed(seed)
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    env    = GridWorld(seed=seed)
    agent  = DQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions,
                      arch=arch, gamma=gamma)
    phi_fn = SHAPING_MAP[shaping]

    window   = deque(maxlen=50)
    lc, ep_losses = [], []
    last200_losses, last200_suc, last200_col = [], [], []
    ce = n_episodes

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        t     = 0
        ep_loss = []

        # Phi at state BEFORE any step
        phi_s = phi_fn(env, t) if phi_fn else 0.0

        while not done:
            action = agent.select_action(state)
            next_state, r_base, done, info = env.step(action)
            t += 1

            if phi_fn:
                phi_s2   = phi_fn(env, t)
                r_shaped = r_base + gamma * phi_s2 - phi_s
                phi_s    = phi_s2
            else:
                r_shaped = r_base

            agent.store(state, action, r_shaped, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)
            state = next_state

        agent.decay_epsilon()

        suc = 1 if info == 'goal'      else 0
        col = 1 if info == 'collision' else 0
        window.append(suc)
        rolling = np.mean(window)
        lc.append(rolling)
        ep_losses.append(np.mean(ep_loss) if ep_loss else 0.0)

        if ce == n_episodes and len(window) == 50 and rolling >= 0.60:
            ce = ep

        if ep >= n_episodes - 200:
            if ep_loss: last200_losses.extend(ep_loss)
            last200_suc.append(suc)
            last200_col.append(col)

    mse = float(np.mean(last200_losses)) if last200_losses else float('nan')
    scr = sum(last200_suc) / sum(last200_col) if sum(last200_col) > 0 \
          else float(sum(last200_suc))

    return ce, np.array(lc), np.array(ep_losses), mse, scr

if __name__ == '__main__':
    print("Testing DQN baseline, seed=1 ...")
    ce, lc, losses, mse, scr = run_one('dqn', 'baseline', seed=1)
    print(f"CE={ce}  MSE={mse:.4f}  SCR={scr:.2f}")
    print(f"Last 5 rolling success: {lc[-5:].round(2)}")