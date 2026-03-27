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

for config in configs:
    ce_list = []

    folder_path = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder_path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder_path, file))
            success = data["success"].tolist()

            ce = compute_ce(success)
            ce_list.append(ce)

    mean_ce = np.mean(ce_list)
    std_ce = np.std(ce_list)

    print(f"{config}: CE = {mean_ce:.2f} ± {std_ce:.2f}")