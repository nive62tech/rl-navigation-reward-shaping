import os
import numpy as np

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
    print(f"\nRunning: {algo} | {shaping} | Seed {seed}")

    # Set seed
    set_seed(seed)

    # Create result folder
    folder = create_result_folder(algo, shaping)

    # ---- PLACEHOLDER DATA (Phase 2 will replace this) ----
    success_history = np.random.randint(0, 2, size=2000).tolist()
    reward_history = np.random.randn(2000).tolist()
    loss_history = np.abs(np.random.randn(5000)).tolist()

    success_eval = np.random.randint(100, 200)
    collision_eval = np.random.randint(1, 50)
    # -----------------------------------------------------

    # Save results
    save_path = os.path.join(folder, f"seed_{seed}.npz")

    np.savez(
        save_path,
        success=success_history,
        reward=reward_history,
        loss=loss_history,
        success_eval=success_eval,
        collision_eval=collision_eval
    )

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