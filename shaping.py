# shaping.py
# Potential-Based Reward Shaping functions.
# F(s,s') = gamma * Phi(s') - Phi(s)
# All potentials are normalized to [-1, 1] range
# to prevent reward explosion.

def phi_mdws(env, t=None, alpha=0.5):
    max_dist = (env.N - 1) * 2        # max possible manhattan distance
    d        = env.manhattan_to_goal()
    return -alpha * (d / max_dist)    # normalized to [-alpha, 0]

def phi_tpds(env, t, T=300, alpha=0.5, beta=0.3):
    max_dist = (env.N - 1) * 2
    d        = env.manhattan_to_goal()
    return -alpha * (d / max_dist) - beta * (t / T)

def phi_ops(env, t=None, alpha=0.5, delta=0.2, D_safe=3):
    max_dist = (env.N - 1) * 2
    d        = env.manhattan_to_goal()
    d_obs    = min(env.manhattan_to_nearest_obstacle(), D_safe)
    return -alpha * (d / max_dist) + delta * (d_obs / D_safe)

SHAPING_MAP = {
    'baseline': None,
    'mdws':     phi_mdws,
    'tpds':     phi_tpds,
    'ops':      phi_ops,
}