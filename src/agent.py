from abc import ABC, abstractmethod
import numpy as np

from src.banana_env import BananaEnv


class Agent(ABC):
    def __init__(self, env: BananaEnv):
        self.env = env
        self.t_step = 0
        self.train_state = None

    @abstractmethod
    def compute_action(self, state, epsilon):
        pass

    def evaluate(self, n=10, fast=True):
        scores = [self.run_episode(fast) for _ in range(n)]
        return np.mean(scores)

    def run_episode(self, fast=False):
        state = self.env.reset(train_mode=fast)
        score = 0
        done = False
        while not done:
            action = self.compute_action(state)
            state, reward, done = self.env.step(action)
            score += reward
            if reward != 0 and not fast:
                print(f"Score {score}")
        return score
