"""
env.py
======
Real 15x15 Grid-World Gymnasium environment.
- Sparse reward: +1 goal, -1 collision, 0 otherwise
- Stochastic transitions (slip probability)
- Full observability: agent pos + goal pos + obstacle map
"""

import numpy as np


class GridWorld:
    def __init__(self, grid_size=7, obstacle_density=0.05,
                 slip_prob=0, max_steps=200, seed=None):
        self.N         = grid_size
        self.rho       = obstacle_density
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.obs_dim   = 3 * self.N * self.N   # 675 for N=15
        self.n_actions = 4                      # Up Down Left Right
        self.start     = (0, 0)
        self.goal      = (self.N - 1, self.N - 1)
        self.rng       = np.random.default_rng(seed)
        self._place_obstacles()

    def _place_obstacles(self):
        """Place obstacles randomly, never on start or goal."""
        n_obs      = int(self.rho * self.N * self.N)
        candidates = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if (r, c) != self.start and (r, c) != self.goal
        ]
        idx            = self.rng.choice(len(candidates),
                                         size=min(n_obs, len(candidates)),
                                         replace=False)
        self.obstacles = {candidates[i] for i in idx}

        # Pre-build obstacle one-hot (same every episode)
        self._obs_onehot = np.zeros(self.N * self.N, dtype=np.float32)
        for r, c in self.obstacles:
            self._obs_onehot[r * self.N + c] = 1.0

    def reset(self):
        self.agent = self.start
        self.steps = 0
        return self._observe()

    def _observe(self):
        r, c   = self.agent
        gr, gc = self.goal

        agent_oh = np.zeros(self.N * self.N, dtype=np.float32)
        goal_oh  = np.zeros(self.N * self.N, dtype=np.float32)
        agent_oh[r * self.N + c]   = 1.0
        goal_oh[gr * self.N + gc]  = 1.0

        return np.concatenate([agent_oh, goal_oh, self._obs_onehot])

    def step(self, action):
        # Stochastic slip: ignore chosen action with prob slip_prob
        if self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(4))

        # Movement deltas: Up, Down, Left, Right
        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][action]
        nr = int(np.clip(self.agent[0] + dr, 0, self.N - 1))
        nc = int(np.clip(self.agent[1] + dc, 0, self.N - 1))
        self.agent = (nr, nc)
        self.steps += 1

        if self.agent == self.goal:
            return self._observe(), +1.0, True, 'goal'
        if self.agent in self.obstacles:
            return self._observe(), -1.0, True, 'collision'
        if self.steps >= self.max_steps:
            return self._observe(), 0.0, True, 'timeout'
        return self._observe(), 0.0, False, 'step'

    # ── Helpers used by shaping ───────────────────────────────────
    def manhattan_to_goal(self):
        return (abs(self.agent[0] - self.goal[0]) +
                abs(self.agent[1] - self.goal[1]))

    def manhattan_to_nearest_obstacle(self):
        if not self.obstacles:
            return self.N * 2
        r, c = self.agent
        return min(abs(r - or_) + abs(c - oc)
                   for or_, oc in self.obstacles)
