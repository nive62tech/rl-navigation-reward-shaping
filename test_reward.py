from src.rewards.reward_shaping import compute_reward

info = {
    "goal_distance": 10,
    "obstacle_distance": 2
}

print("Baseline:", compute_reward(1.0, "baseline", info))
print("MDWS:", compute_reward(1.0, "mdws", info))
print("TPDS:", compute_reward(1.0, "tpds", info))
print("OPS:", compute_reward(1.0, "ops", info))