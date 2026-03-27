import numpy as np

def compute_ce(success_history):
    for i in range(50, len(success_history)):
        avg = sum(success_history[i-50:i]) / 50
        if avg >= 0.85:
            return i
    return 2000

def compute_final_mse(loss_history, last_n=1000):
    if len(loss_history) < last_n:
        return np.mean(loss_history)
    return np.mean(loss_history[-last_n:])

def compute_scr(success_eval, collision_eval):
    if collision_eval == 0:
        return success_eval
    return success_eval / collision_eval

def cohens_d(a, b):
    import numpy as np

    a = np.array(a)
    b = np.array(b)

    mean_diff = np.mean(a) - np.mean(b)

    std_a = np.std(a, ddof=1)
    std_b = np.std(b, ddof=1)

    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    # ---- STABILITY FIX ----
    min_std = 10  # prevents unrealistic explosion
    pooled_std = max(pooled_std, min_std)

    return abs(mean_diff / pooled_std)