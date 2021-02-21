import numpy as np

from src.agent import Agent
from src.banana_env import BananaEnv


class RandomAgent(Agent):
    def __init__(self, env: BananaEnv):
        super().__init__(env)

    def compute_action(self, state):
        return np.random.randint(self.env.action_size)
