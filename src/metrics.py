from collections import deque
import numpy as np


class Metrics:
    def __init__(self):
        self.step_count = 0
        self.episode_count = 0
        self.score_window = deque([0], maxlen=100)
        self.score = 0
        self.current_episode_length = 0

    def on_step(self, reward, done):
        self.step_count += 1
        self.current_episode_length += 1
        self.score += reward
        if done:
            self.episode_count += 1
            self.score_window.append(self.score)
            self.score = 0
            self.current_episode_length = 0

    def running_score(self):
        return np.mean(self.score_window)
