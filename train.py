import sys
import torch

from src.banana_env import BananaEnv
from src.dqn_agent import DQNAgent, DQNConfig


def get_env_path():
    if len(sys.argv) != 2:
        print("ERROR: invalid arguments list")
        print("Usage: train.py <path_to_unity_env>")
        sys.exit(1)
    return sys.argv[1]


if __name__ == '__main__':
    env = BananaEnv(get_env_path())
    agent = DQNAgent(env, DQNConfig())
    agent.train(1000000, 4, 1000)
    print(agent.evaluate())
    env.close()
