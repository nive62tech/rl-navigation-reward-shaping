import os
import numpy as np
from scipy.stats import mannwhitneyu
from src.utils.metrics import compute_ce

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

# ---- Collect CE per seed ----
ce_data = {}

for config in configs:
    ce_list = []
    folder = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder, file))
            success = data["success"].tolist()
            ce = compute_ce(success)
            ce_list.append(ce)

    ce_data[config] = ce_list

# ---- Baseline ----
baseline = ce_data["DQN_baseline"]

# ---- Bonferroni threshold ----
threshold = 0.05 / 7

print(f"\nBonferroni threshold: {threshold:.4f}\n")

# ---- Statistical testing ----
for config in configs:
    if config == "DQN_baseline":
        continue

    stat, p = mannwhitneyu(ce_data[config], baseline, alternative='less')

    significance = "SIGNIFICANT" if p < threshold else "NOT SIGNIFICANT"

    print(f"{config}: p = {p:.5f} → {significance}")