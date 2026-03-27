# src/rewards/reward_shaping.py

ALPHA = 0.01
BETA = 0.001
GAMMA = 0.1

def compute_reward(base_reward, shaping_type, info):
    goal_dist = info.get("goal_distance", 0.0)
    obs_dist = info.get("obstacle_distance", 1.0)

    if shaping_type == "mdws":
        return base_reward - ALPHA * goal_dist

    elif shaping_type == "tpds":
        return base_reward - BETA

    elif shaping_type == "ops":
        return base_reward - GAMMA / (obs_dist + 1e-5)

    return base_reward