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

