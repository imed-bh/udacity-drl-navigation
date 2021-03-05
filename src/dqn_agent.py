import torch
import torch.nn.functional as F
import numpy as np

from src.agent import Agent
from src.metrics import Metrics
from src.replay import ReplayBuffer, Experience
from src.qnetwork import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNConfig:
    def __init__(self,
                 buffer_size=100000,
                 batch_size=256,
                 learning_rate=0.0001,
                 tau=0.001,
                 gamma=0.99,
                 fc1_units=128,
                 fc2_units=128,
                 episode_max_length=300):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.episode_max_length = episode_max_length


class DQNAgent(Agent):
    def __init__(self, env, config: DQNConfig):
        super().__init__(env)
        self.config = config
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.batch_size)
        self.qnet_current = QNetwork(env.state_size, env.action_size, config.fc1_units, config.fc2_units).to(device)
        self.qnet_target = QNetwork(env.state_size, env.action_size, config.fc1_units, config.fc2_units).to(device)
        self.optimizer = torch.optim.Adam(self.qnet_current.parameters(), lr=config.learning_rate)
        self.metrics = Metrics()

    def restore(self, file):
        self.qnet_current.load_state_dict(torch.load(file))

    def compute_action(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_size)

        action_values = self.qnet_current.action_values_for(state)
        return np.argmax(action_values)

    def train(self, n_steps, update_every, print_every, epsilon_init=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        epsilon = epsilon_init
        state = self._warmup(epsilon)
        self.metrics.plot()

        for t_step in range(1, n_steps + 1):
            state = self._step(state, epsilon)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if t_step % update_every == 0:
                self._batch_train()
                if self._check_solved():
                    break

            if t_step % print_every == 0:
                print(f"Step #{t_step}" +
                      f", Running score {self.metrics.running_score():.2f}" +
                      f", Total steps {self.metrics.step_count}" +
                      f", Total episodes {self.metrics.episode_count}")

    def _warmup(self, epsilon):
        state = self.env.reset(train_mode=True)
        needed_experiences = max(0, self.replay_buffer.batch_size - len(self.replay_buffer))
        for i in range(needed_experiences):
            state = self._step(state, epsilon)
        return state

    def _step(self, state, epsilon):
        action = self.compute_action(state, epsilon)
        next_state, reward, done = self.env.step(action)
        self.replay_buffer.add(Experience(state, action, reward, next_state, done))
        if self.metrics.current_episode_length >= self.config.episode_max_length:
            done = True
        self.metrics.on_step(reward, done)
        if done:
            return self.env.reset(train_mode=True)
        return next_state

    def _batch_train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnet_current(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.qnet_target.soft_update(self.qnet_current, self.config.tau)

    def _check_solved(self):
        if self.metrics.running_score() >= 13:
            print(f"\nEnvironment solved in {self.metrics.episode_count} episodes!\t" +
                  f"Average Score: {self.metrics.running_score():.2f}")
            torch.save(self.qnet_current.state_dict(), "model.pt")
            return True

        return False
