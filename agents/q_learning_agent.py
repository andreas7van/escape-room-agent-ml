import random
import pickle

import numpy as np


class QLearningAgent:
    """
    Q-Learning agent για το Escape Room.

    Ο agent χρησιμοποιεί Q-table:
        Q[state][action] = value

    Επιλέγει ενέργειες με epsilon-greedy στρατηγική.
    """

    def __init__(
        self,
        action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
    ):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        return self.q_table[state]

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)

        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        current_q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        current_q = current_q_values[action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(next_q_values)

        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, filepath):
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "action_space_size": self.action_space_size,
        }

        with open(filepath, "wb") as file:
            pickle.dump(data, file)

    def load(self, filepath):
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.learning_rate = data["learning_rate"]
        self.discount_factor = data["discount_factor"]
        self.epsilon_decay = data["epsilon_decay"]
        self.min_epsilon = data["min_epsilon"]
        self.action_space_size = data["action_space_size"]
