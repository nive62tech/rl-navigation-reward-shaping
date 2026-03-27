import os
import numpy as np
from src.train import train

# =========================
# CONFIGURATIONS
# =========================
configs = [
    ("DQN", "baseline"),
    ("DQN", "mdws"),
    ("DQN", "tpds"),
    ("DQN", "ops"),
    ("DDQN", "baseline"),
    ("DDQN", "mdws"),
    ("DDQN", "tpds"),
    ("DDQN", "ops"),
]

seeds = [1, 2, 3, 4, 5]


# =========================
# SEED CONTROL
# =========================
def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================
# PATH HANDLING (FIXED)
# =========================
# Always point to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def create_result_folder(algo, shaping):
    folder_name = os.path.join(RESULTS_DIR, f"{algo}_{shaping}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


# =========================
# EXPERIMENT FUNCTION
# =========================
def run_experiment(algo, shaping, seed):
    print(f"Running: {algo} | {shaping} | Seed {seed}")

    set_seed(seed)

    # ✅ ADD THIS BACK
    folder = create_result_folder(algo, shaping)

    # ✅ TRAIN
    logger = train(algo, shaping, seed)

    # ✅ SAVE
    save_path = os.path.join(folder, f"seed_{seed}.npz")
    logger.save(save_path)

    print(f"Saved: {save_path}")


# =========================
# MAIN LOOP
# =========================
def main():
    print("Project Root:", BASE_DIR)
    print("Results Directory:", RESULTS_DIR)

    for algo, shaping in configs:
        for seed in seeds:
            run_experiment(algo, shaping, seed)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()