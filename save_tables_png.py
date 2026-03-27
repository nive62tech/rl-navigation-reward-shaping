import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics import compute_ce, compute_final_mse, compute_scr

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

# ---- collect data ----
ce_mean, ce_std = {}, {}
mse_mean, mse_std = {}, {}
scr_mean, scr_std = {}, {}

for config in configs:
    ce_list, mse_list, scr_list = [], [], []
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

    ce_mean[config] = np.mean(ce_list)
    ce_std[config] = np.std(ce_list)

    mse_mean[config] = np.mean(mse_list)
    mse_std[config] = np.std(mse_list)

    scr_mean[config] = np.mean(scr_list)
    scr_std[config] = np.std(scr_list)

# ---- delta ----
baseline_ce = ce_mean["DQN_baseline"]
delta = {c: ((baseline_ce - ce_mean[c]) / baseline_ce) * 100 if c != "DQN_baseline" else 0 for c in configs}

# =========================
# 📊 TABLE II (CE + Δ%)
# =========================
table2_data = []
for c in configs:
    table2_data.append([
        c,
        f"{ce_mean[c]:.2f} ± {ce_std[c]:.2f}",
        f"{delta[c]:.2f}%"
    ])

fig, ax = plt.subplots(figsize=(10,4))
ax.axis('off')
table = ax.table(
    cellText=table2_data,
    colLabels=["Config", "CE (Mean ± SD)", "Δ%"],
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.savefig("table_II.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 📊 TABLE III (MSE + SCR)
# =========================
table3_data = []
for c in configs:
    table3_data.append([
        c,
        f"{mse_mean[c]:.4f} ± {mse_std[c]:.4f}",
        f"{scr_mean[c]:.2f} ± {scr_std[c]:.2f}"
    ])

fig, ax = plt.subplots(figsize=(10,4))
ax.axis('off')
table = ax.table(
    cellText=table3_data,
    colLabels=["Config", "MSE (Mean ± SD)", "SCR (Mean ± SD)"],
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.savefig("table_III.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved: table_II.png and table_III.png")