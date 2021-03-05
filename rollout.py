import sys

from src.banana_env import BananaEnv
from src.dqn_agent import DQNAgent, DQNConfig
from src.random_agent import RandomAgent


# choose agent: dqn | random
AGENT = "dqn"


def get_env_path():
    if len(sys.argv) != 2:
        print("ERROR: invalid arguments list")
        print("Usage: rollout.py <path_to_unity_env>")
        sys.exit(1)
    return sys.argv[1]


if __name__ == "__main__":
    env = BananaEnv(get_env_path())
    agent = RandomAgent(env) if AGENT == "random" else DQNAgent(env, DQNConfig())
    agent.restore("model.pt")
    agent.run_episode()
    env.close()
