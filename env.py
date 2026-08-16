import numpy as np
import random

class GridWorld:
    def __init__(self, grid_size=10, obstacle_density=0.10,
                 slip_prob=0.0, max_steps=300, seed=None):
        self.N         = grid_size
        self.rho       = obstacle_density
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.obs_dim   = 6
        self.n_actions = 4
        self.start     = (0, 0)
        self.goal      = (self.N - 1, self.N - 1)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._place_obstacles()

    def _place_obstacles(self):
        n_obs      = int(self.rho * self.N * self.N)
        candidates = [(r,c) for r in range(self.N) for c in range(self.N)
                      if (r,c) != self.start and (r,c) != self.goal]
        self.obstacles = set(random.sample(candidates, min(n_obs, len(candidates))))

    def reset(self):
        self.agent = list(self.start)
        self.steps = 0
        return self._obs()

    def _obs(self):
        r,c    = self.agent
        gr,gc  = self.goal
        N      = self.N
        return np.array([r/N, c/N, gr/N, gc/N,
                         (gr-r)/N, (gc-c)/N], dtype=np.float32)

    def step(self, action):
        if self.slip_prob > 0 and random.random() < self.slip_prob:
            action = random.randint(0,3)
        r,c = self.agent
        if   action == 0: r = max(0,         r-1)
        elif action == 1: r = min(self.N-1,  r+1)
        elif action == 2: c = max(0,         c-1)
        elif action == 3: c = min(self.N-1,  c+1)
        self.agent = [r,c]
        self.steps += 1
        if (r,c) == self.goal:      return self._obs(), +1.0, True,  'goal'
        if (r,c) in self.obstacles: return self._obs(), -1.0, True,  'collision'
        if self.steps >= self.max_steps: return self._obs(), 0.0, True, 'timeout'
        return self._obs(), 0.0, False, 'step'

    def manhattan_to_goal(self):
        r,c=self.agent; gr,gc=self.goal
        return abs(r-gr)+abs(c-gc)

    def manhattan_to_nearest_obstacle(self):
        if not self.obstacles: return self.N*2
        r,c=self.agent
        return min(abs(r-or_)+abs(c-oc) for or_,oc in self.obstacles)