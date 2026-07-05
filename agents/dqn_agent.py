import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """
    Νευρωνικό δίκτυο για Deep Q-Learning.

    Input:
        state vector

    Output:
        Q-value για κάθε πιθανή ενέργεια
    """

    def __init__(self, state_size, action_space_size):
        super(DQNNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size),
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    """
    Experience Replay Buffer.

    Αποθηκεύει εμπειρίες της μορφής:
        state, action, reward, next_state, done
    """

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent.

    Χρησιμοποιεί:
    - policy network
    - target network
    - experience replay
    - epsilon-greedy exploration
    """

    def __init__(
        self,
        state_size,
        action_space_size,
        grid_rows=6,
        grid_cols=6,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        batch_size=64,
        replay_buffer_size=50000,
        device=None,
    ):
        self.state_size = state_size
        self.action_space_size = action_space_size

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

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

        self.policy_network = DQNNetwork(
            state_size=state_size,
            action_space_size=action_space_size,
        ).to(self.device)

        self.target_network = DQNNetwork(
            state_size=state_size,
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
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    def normalize_state(self, state):
        """
        Μετατρέπει το state tuple σε αριθμητικό vector.
        Κανονικοποιεί row/col ώστε να βοηθηθεί το νευρωνικό δίκτυο.
        """

        state_array = np.array(state, dtype=np.float32)

        state_array[0] = state_array[0] / max(1, self.grid_rows - 1)
        state_array[1] = state_array[1] / max(1, self.grid_cols - 1)

        return state_array

    def choose_action(self, state, training=True):
        state_array = self.normalize_state(state)

        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)

        state_tensor = torch.tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        state_array = self.normalize_state(state)
        next_state_array = self.normalize_state(next_state)

        self.replay_buffer.add(
            state_array,
            action,
            reward,
            next_state_array,
            done,
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
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

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        current_q_values = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
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
            "state_size": self.state_size,
            "action_space_size": self.action_space_size,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
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

        self.grid_rows = checkpoint.get("grid_rows", self.grid_rows)
        self.grid_cols = checkpoint.get("grid_cols", self.grid_cols)

        self.policy_network.eval()
        self.target_network.eval()
