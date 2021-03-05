import sys

from src.banana_env import BananaEnv
from src.dqn_agent import DQNAgent, DQNConfig

BUFFER_SIZE = 100000
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
TAU = 0.001
GAMMA = 0.99
FC1_UNITS = 128
FC2_UNITS = 128
EPISODE_MAX_LENGTH = 300
N_STEPS = 200000
UPDATE_EVERY = 4
PRINT_EVERY = 1000
EPS_INIT = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.01


def get_env_path():
    if len(sys.argv) != 2:
        print("ERROR: invalid arguments list")
        print("Usage: train.py <path_to_unity_env>")
        sys.exit(1)
    return sys.argv[1]


if __name__ == '__main__':
    env = BananaEnv(get_env_path())
    agent = DQNAgent(env, DQNConfig(
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        tau=TAU,
        gamma=GAMMA,
        fc1_units=FC1_UNITS,
        fc2_units=FC2_UNITS,
        episode_max_length=EPISODE_MAX_LENGTH,
    ))
    agent.train(N_STEPS, UPDATE_EVERY, PRINT_EVERY, EPS_INIT, EPS_DECAY, EPS_MIN)
    input('Press key to continue...')
    env.close()
