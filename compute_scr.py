import os
import numpy as np
from src.utils.metrics import compute_scr

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
    scr_list = []

    folder_path = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder_path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder_path, file))

            success_eval = int(data["success_eval"])
            collision_eval = int(data["collision_eval"])

            scr = compute_scr(success_eval, collision_eval)
            scr_list.append(scr)

    mean_scr = np.mean(scr_list)
    std_scr = np.std(scr_list)

    print(f"{config}: SCR = {mean_scr:.2f} ± {std_scr:.2f}")