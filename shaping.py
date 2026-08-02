"""
shaping.py
==========
Potential-Based Reward Shaping functions.
All satisfy F(s,s') = gamma * Phi(s') - Phi(s).
This guarantees policy invariance (Ng et al., 1999).
"""


def phi_mdws(env, t=None, alpha=0.5):
    """Manhattan Distance-Weighted Shaping."""
    return -alpha * env.manhattan_to_goal()


def phi_tpds(env, t, T=500, alpha=0.5, beta=0.3):
    """Time-Penalised Distance Shaping."""
    return -alpha * env.manhattan_to_goal() - beta * (t / T)


def phi_ops(env, t=None, alpha=0.5, delta=0.2, D_safe=3):
    """Obstacle-Proximity Shaping."""
    return (-alpha * env.manhattan_to_goal() +
            delta * min(env.manhattan_to_nearest_obstacle(), D_safe))


def get_shaped_reward(r_base, phi_fn, env, t, gamma=0.99,
                      phi_prev=None):
    """
    Compute shaped reward:
        r' = r_base + gamma * Phi(s') - Phi(s)

    phi_prev must be Phi(s) computed BEFORE the step.
    Returns (r_shaped, phi_next) so phi_next can be used
    as phi_prev in the next step.
    """
    if phi_fn is None:
        return r_base, 0.0

    phi_next = phi_fn(env, t)
    r_shaped = r_base + gamma * phi_next - phi_prev
    return r_shaped, phi_next


# Map shaping name → phi function (None = no shaping)
SHAPING_MAP = {
    'baseline': None,
    'mdws':     phi_mdws,
    'tpds':     phi_tpds,
    'ops':      phi_ops,
}
