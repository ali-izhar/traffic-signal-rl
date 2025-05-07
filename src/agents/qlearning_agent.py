#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import pickle

from .base import Agent


class QLearningAgent(Agent):
    """Tabular Q-Learning Agent implementation"""

    def __init__(
        self, input_dim, output_dim, learning_rate=0.1, gamma=0.75, initial_value=0.0
    ):
        super().__init__(input_dim, output_dim)
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._initial_value = initial_value

        # Since states in traffic environment are continuous (or very large discrete space),
        # we'll need to discretize the state space for tabular Q-learning
        self._q_table = {}  # Using a sparse representation with dictionary

    def _get_state_key(self, state):
        """Convert state array to a hashable key by thresholding values"""
        # Convert continuous/high-dimensional state to a discrete representation
        # For traffic simulation, we'll consider a cell as occupied (1) or not (0)
        return tuple(1 if x > 0.5 else 0 for x in state)

    def act(self, state, epsilon=0):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self._output_dim - 1)  # Explore
        else:
            return self._get_best_action(state)  # Exploit

    def _get_best_action(self, state):
        """Get best action for the given state based on Q-values"""
        state_key = self._get_state_key(state)

        # If state not in Q-table, initialize it
        if state_key not in self._q_table:
            self._q_table[state_key] = [self._initial_value] * self._output_dim

        return np.argmax(self._q_table[state_key])

    def learn(self, state, action, reward, next_state, done=False):
        """Update Q-table using the Q-learning update rule"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Initialize Q-values if not in table
        if state_key not in self._q_table:
            self._q_table[state_key] = [self._initial_value] * self._output_dim

        if next_state_key not in self._q_table:
            self._q_table[next_state_key] = [self._initial_value] * self._output_dim

        # Q-learning update rule
        if done:
            # If terminal state, no future reward
            target = reward
        else:
            # Standard Q-learning (off-policy TD learning)
            target = reward + self._gamma * max(self._q_table[next_state_key])

        # Update Q-value
        current_q = self._q_table[state_key][action]
        self._q_table[state_key][action] = current_q + self._learning_rate * (
            target - current_q
        )

    def save(self, path):
        """Save Q-table to disk"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "q_table.pkl"), "wb") as f:
            pickle.dump(self._q_table, f)

        # Save parameters separately for reference
        params = {
            "learning_rate": self._learning_rate,
            "gamma": self._gamma,
            "initial_value": self._initial_value,
        }
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump(params, f)

    def load(self, path):
        """Load Q-table from disk"""
        q_table_path = os.path.join(path, "q_table.pkl")
        params_path = os.path.join(path, "params.pkl")

        if os.path.isfile(q_table_path):
            with open(q_table_path, "rb") as f:
                self._q_table = pickle.load(f)
        else:
            raise FileNotFoundError(f"No Q-table found at {q_table_path}")

        if os.path.isfile(params_path):
            with open(params_path, "rb") as f:
                params = pickle.load(f)
                self._learning_rate = params.get("learning_rate", self._learning_rate)
                self._gamma = params.get("gamma", self._gamma)
                self._initial_value = params.get("initial_value", self._initial_value)
