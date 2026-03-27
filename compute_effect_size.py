import os
import numpy as np
from src.utils.metrics import compute_ce, compute_final_mse, compute_scr, cohens_d

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

# ---- Collect data ----
ce_data = {}
mse_data = {}
scr_data = {}

for config in configs:
    ce_list = []
    mse_list = []
    scr_list = []

    folder = os.path.join(RESULTS_DIR, config)

    for file in os.listdir(folder):
        if file.endswith(".npz"):
            data = np.load(os.path.join(folder, file))

            success = data["success"].tolist()
            loss = data["loss"].tolist()
            success_eval = int(data["success_eval"])
            collision_eval = int(data["collision_eval"])

            ce_list.append(compute_ce(success))
            mse_list.append(compute_final_mse(loss))
            scr_list.append(compute_scr(success_eval, collision_eval))

    ce_data[config] = ce_list
    mse_data[config] = mse_list
    scr_data[config] = scr_list

# ---- Baseline ----
baseline_ce = ce_data["DQN_baseline"]
baseline_mse = mse_data["DQN_baseline"]
baseline_scr = scr_data["DQN_baseline"]

print("\nEffect Sizes (Cohen's d vs DQN_baseline)\n")

for config in configs:
    if config == "DQN_baseline":
        continue

    d_ce = cohens_d(ce_data[config], baseline_ce)
    d_mse = cohens_d(mse_data[config], baseline_mse)
    d_scr = cohens_d(scr_data[config], baseline_scr)

    print(f"{config}: CE={d_ce:.2f}, MSE={d_mse:.2f}, SCR={d_scr:.2f}")