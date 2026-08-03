"""
env.py - Fixed Grid-World with coordinate-based observation
Much easier for the network to learn from than one-hot encoding.
Observation: [agent_row/N, agent_col/N, goal_row/N, goal_col/N,
              delta_row/N, delta_col/N] = 6 values
"""
import numpy as np


class GridWorld:
    def __init__(self, grid_size=10, obstacle_density=0.10,
                 slip_prob=0.0, max_steps=300, seed=None):
        self.N         = grid_size
        self.rho       = obstacle_density
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.obs_dim   = 6          # normalized coordinates only
        self.n_actions = 4
        self.start     = (0, 0)
        self.goal      = (self.N - 1, self.N - 1)
        self.rng       = np.random.default_rng(seed)
        self._place_obstacles()

    def _place_obstacles(self):
        n_obs      = int(self.rho * self.N * self.N)
        candidates = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if (r, c) != self.start and (r, c) != self.goal
        ]
        if n_obs > 0 and candidates:
            idx = self.rng.choice(len(candidates),
                                  size=min(n_obs, len(candidates)),
                                  replace=False)
            self.obstacles = {candidates[i] for i in idx}
        else:
            self.obstacles = set()

    def reset(self):
        self.agent = self.start
        self.steps = 0
        return self._observe()

    def _observe(self):
        r,  c  = self.agent
        gr, gc = self.goal
        N = self.N
        return np.array([
            r  / N,
            c  / N,
            gr / N,
            gc / N,
            (gr - r) / N,
            (gc - c) / N,
        ], dtype=np.float32)

    def step(self, action):
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(4))

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

    def manhattan_to_goal(self):
        return (abs(self.agent[0] - self.goal[0]) +
                abs(self.agent[1] - self.goal[1]))

    def manhattan_to_nearest_obstacle(self):
        if not self.obstacles:
            return self.N * 2
        r, c = self.agent
        return min(abs(r - or_) + abs(c - oc)
                   for or_, oc in self.obstacles)