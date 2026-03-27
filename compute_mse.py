import os
import numpy as np
from src.utils.metrics import compute_final_mse

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
    mse_list = []

    folder_path = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder_path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder_path, file))
            loss = data["loss"].tolist()

            mse = compute_final_mse(loss)
            mse_list.append(mse)

    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    print(f"{config}: MSE = {mean_mse:.4f} ± {std_mse:.4f}")