import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.observation import NUM_CHANNELS, OBS_SIZE


class CNNDuelingNetwork(nn.Module):
    """
    Convolutional dueling Q-network over spatial observations.

    Input:
        (batch, NUM_CHANNELS, OBS_SIZE, OBS_SIZE) observation tensor

    Output:
        Q-value per action, combined from separate state-value and
        advantage streams (dueling architecture).
    """

    def __init__(self, action_space_size, num_channels=NUM_CHANNELS, obs_size=OBS_SIZE):
        super(CNNDuelingNetwork, self).__init__()

        # Two convolutions at full grid resolution, then a 2x2 max-pool
        # before the last one: this keeps per-cell detail where it matters
        # while making CPU training ~4x faster than a full-resolution stack.
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        feature_size = 64 * (obs_size // 2) * (obs_size // 2)

        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size),
        )

    def forward(self, x):
        features = self.features(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ObservationReplayBuffer:
    """
    Experience replay buffer for spatial observations.

    Observations are stored as float16 to halve memory use; they are cast
    back to float32 when a batch is sampled.
    """

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, observation, action, reward, next_observation, done):
        self.buffer.append(
            (
                observation.astype(np.float16),
                action,
                reward,
                next_observation.astype(np.float16),
                done,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)

        return (
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_observations, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class CNNDQNAgent:
    """
    Double DQN agent with a convolutional dueling network.

    Differences from the original DQNAgent (which stays untouched):
    - takes spatial (C, H, W) observations instead of state tuples, so one
      network generalizes across level layouts and sizes
    - Double DQN targets: the policy network selects the next action, the
      target network evaluates it (reduces Q-value overestimation)
    - dueling value/advantage heads
    """

    def __init__(
        self,
        action_space_size,
        learning_rate=0.0005,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.05,
        batch_size=64,
        replay_buffer_size=50000,
        device=None,
    ):
        self.action_space_size = action_space_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.batch_size = batch_size

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.policy_network = CNNDuelingNetwork(
            action_space_size=action_space_size,
        ).to(self.device)

        self.target_network = CNNDuelingNetwork(
            action_space_size=action_space_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ObservationReplayBuffer(max_size=replay_buffer_size)

    # When exploring, favour movement (actions 0-3) over interactions
    # (PICK_KEY / SOLVE_PUZZLE / OPEN_DOOR): uniform sampling wastes almost
    # half the exploratory steps on interactions that only make sense on
    # specific cells.
    MOVEMENT_EXPLORATION_BIAS = 0.75

    def choose_action(self, observation, training=True):
        if training and random.random() < self.epsilon:
            if random.random() < self.MOVEMENT_EXPLORATION_BIAS:
                return random.randint(0, 3)
            return random.randint(4, self.action_space_size - 1)

        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(observation_tensor)

        return int(torch.argmax(q_values).item())

    def remember(self, observation, action, reward, next_observation, done):
        self.replay_buffer.add(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        observations, actions, rewards, next_observations, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        observations = torch.tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_observations = torch.tensor(
            next_observations,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        current_q_values = self.policy_network(observations).gather(1, actions)

        with torch.no_grad():
            # Double DQN: policy network picks the action, target network
            # evaluates it.
            next_actions = self.policy_network(next_observations).argmax(
                dim=1, keepdim=True
            )
            next_q_values = self.target_network(next_observations).gather(
                1, next_actions
            )
            target_q_values = rewards + (
                self.discount_factor * next_q_values * (1 - dones)
            )

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, filepath):
        checkpoint = {
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "batch_size": self.batch_size,
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.policy_network.eval()
        self.target_network.eval()
