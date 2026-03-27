import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

configs = [
    "DQN_baseline",
    "DQN_mdws",
    "DQN_tpds",
    "DQN_ops",
    "DDQN_baseline",
    "DDQN_mdws",
    "DDQN_tpds",
    "DDQN_ops",
]

# ---- smoothing function ----
def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ---- aggregate rewards & loss ----
reward_data = {}
loss_data = {}

for config in configs:
    rewards_all = []
    losses_all = []

    folder = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder, file))

            rewards_all.append(data["reward"])
            losses_all.append(data["loss"])

    # average across seeds
    reward_data[config] = np.mean(rewards_all, axis=0)
    loss_data[config] = np.mean(losses_all, axis=0)

# ==============================
# 📈 FIGURE 1 — LEARNING CURVES
# ==============================

plt.figure(figsize=(10,6))

for config in configs:
    smoothed = smooth(reward_data[config])
    plt.plot(smoothed, label=config)

plt.xlabel("Episode")
plt.ylabel("Reward (50-episode avg)")
plt.title("Learning Curves")
plt.legend()
plt.grid()

plt.savefig("figure_learning_curves.png", dpi=300)
plt.close()

# ==============================
# 📉 FIGURE 2 — LOSS CURVES
# ==============================

plt.figure(figsize=(10,6))

for config in configs:
    smoothed = smooth(loss_data[config], window=100)
    plt.plot(smoothed, label=config)

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("MSE Loss Curves")
plt.legend()
plt.grid()

plt.savefig("figure_loss_curves.png", dpi=300)
plt.close()

print("Figures saved: figure_learning_curves.png & figure_loss_curves.png")