import os
import numpy as np
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

# ---- Step 1: compute CE for all configs ----
ce_results = {}

for config in configs:
    ce_list = []
    folder_path = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder_path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder_path, file))
            success = data["success"].tolist()

            ce = compute_ce(success)
            ce_list.append(ce)

    ce_results[config] = np.mean(ce_list)

# ---- Step 2: baseline ----
baseline_ce = ce_results["DQN_baseline"]

# ---- Step 3: compute delta ----
for config in configs:
    if config == "DQN_baseline":
        continue

    ce = ce_results[config]

    delta = ((baseline_ce - ce) / baseline_ce) * 100

    print(f"{config}: Δ% = {delta:.2f}%")