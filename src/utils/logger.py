import os
import numpy as np

class Logger:
    def __init__(self):
        self.success_history = []
        self.reward_history = []
        self.loss_history = []

        self.success_eval = 0
        self.collision_eval = 0

    def log_episode(self, success, reward):
        self.success_history.append(success)
        self.reward_history.append(reward)

    def log_loss(self, loss):
        self.loss_history.append(loss)

    def log_eval(self, success, collision):
        self.success_eval += success
        self.collision_eval += collision

    def save(self, save_path):
        np.savez(
            save_path,
            success=self.success_history,
            reward=self.reward_history,
            loss=self.loss_history,
            success_eval=self.success_eval,
            collision_eval=self.collision_eval
        )