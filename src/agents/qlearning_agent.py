#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tabular Q-Learning Agent

This module implements a basic tabular Q-learning agent for traffic signal control,
providing a baseline for comparison with deep RL methods.
"""

import os
import numpy as np
import pickle
from typing import Tuple, Dict, List, Any, Union


class QLearningAgent:
    """
    Tabular Q-learning agent for traffic signal control.

    This agent discretizes the continuous state space and uses a table to store
    Q-values for state-action pairs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        discretization_levels: Union[int, List[int]] = 5,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialize the Q-learning agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for Q-value updates
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            discretization_levels: Number of discretization levels per state dimension
                                 (can be a single integer or a list for custom levels per dimension)
            checkpoint_dir: Directory to save/load Q-table
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.checkpoint_dir = checkpoint_dir

        # Discretization setup
        if isinstance(discretization_levels, int):
            self.discretization_levels = [discretization_levels] * state_dim
        else:
            assert (
                len(discretization_levels) == state_dim
            ), "Discretization levels must match state dimensions"
            self.discretization_levels = discretization_levels

        # Create bounds for each state dimension (will be updated during learning)
        self.state_mins = (
            np.zeros(state_dim) - 0.1
        )  # Small offset to prevent edge issues
        self.state_maxs = np.ones(state_dim) * 10.1  # Initial guess, will adapt

        # Pre-define discretization bins for common state dimensions based on domain knowledge
        # Queue lengths typically range from 0 to max_queue_length (30 in config)
        # Waiting times range from 0 to max_wait_time (100 in config)
        # Densities are normalized to [0,1]
        # Phase is discrete {0,1,2,3}
        # Phase time ranges widely but higher precision needed for small values

        if state_dim >= 14:  # Based on the standard state structure in the paper
            self.state_mins = np.array(
                [
                    0,
                    0,
                    0,
                    0,  # Queue lengths (N,S,E,W)
                    0,
                    0,
                    0,
                    0,  # Waiting times (N,S,E,W)
                    0,
                    0,
                    0,
                    0,  # Densities (N,S,E,W)
                    0,
                    0,  # Phase, phase time
                ]
            )[:state_dim]

            self.state_maxs = np.array(
                [
                    30,
                    30,
                    30,
                    30,  # Queue lengths max
                    100,
                    100,
                    100,
                    100,  # Waiting times max
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # Densities max
                    3,
                    30,  # Phase max, phase time max (for discretization)
                ]
            )[:state_dim]

            # Custom discretization levels for different state dimensions
            self.discretization_levels = [
                5,
                5,
                5,
                5,  # Queue lengths (coarser)
                4,
                4,
                4,
                4,  # Waiting times (coarser)
                3,
                3,
                3,
                3,  # Densities (coarser)
                4,
                6,  # Phase (exact), phase time (finer)
            ][:state_dim]

        # Initialize Q-table with small random values
        self.q_table = {}

        # Create directory for saving checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training metrics
        self.training_steps = 0
        self.episode_count = 0

    def discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize a continuous state into a discrete state usable as Q-table key.

        Args:
            state: Continuous state vector

        Returns:
            Tuple representation of discretized state
        """
        # Update state bounds based on observations
        self.state_mins = np.minimum(self.state_mins, state)
        self.state_maxs = np.maximum(self.state_maxs, state)

        # Discretize each dimension
        discrete_state = []
        for i, (s, s_min, s_max, levels) in enumerate(
            zip(state, self.state_mins, self.state_maxs, self.discretization_levels)
        ):
            # For phase (typically dimension 12 in state vector), use the exact value as is
            if i == 12 and self.state_dim >= 13:
                discrete_state.append(int(s))
            else:
                # Create discretization bins
                if s_max > s_min:
                    bin_width = (s_max - s_min) / levels
                    bin_idx = min(int((s - s_min) // bin_width), levels - 1)
                    bin_idx = max(0, bin_idx)  # Ensure non-negative
                    discrete_state.append(bin_idx)
                else:
                    discrete_state.append(0)

        return tuple(discrete_state)

    def act(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Discretize state
        discrete_state = self.discretize_state(state)

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # Exploitation
        if discrete_state not in self.q_table:
            # Initialize with small random values if state not seen before
            self.q_table[discrete_state] = np.random.uniform(
                low=0, high=0.1, size=self.action_dim
            )

        return np.argmax(self.q_table[discrete_state])

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """
        Update Q-value based on observed transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done

        Returns:
            Dictionary with learning metrics
        """
        # Discretize states
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        # Initialize Q-values if states not seen before
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.random.uniform(
                low=0, high=0.1, size=self.action_dim
            )

        if discrete_next_state not in self.q_table and not done:
            self.q_table[discrete_next_state] = np.random.uniform(
                low=0, high=0.1, size=self.action_dim
            )

        # Get current Q-value
        current_q = self.q_table[discrete_state][action]

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[discrete_next_state])

        # Update Q-value
        td_error = target_q - current_q
        self.q_table[discrete_state][action] += self.lr * td_error

        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update training statistics
        self.training_steps += 1

        if done:
            self.episode_count += 1

        return {
            "td_error": td_error,
            "epsilon": self.epsilon,
            "q_value": current_q,
            "q_table_size": len(self.q_table),
        }

    def save(self, filepath: str) -> None:
        """
        Save Q-table and agent parameters.

        Args:
            filepath: Path to save agent state
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save agent state
        state = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "state_mins": self.state_mins,
            "state_maxs": self.state_maxs,
            "training_steps": self.training_steps,
            "episode_count": self.episode_count,
            "discretization_levels": self.discretization_levels,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"Agent saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load Q-table and agent parameters.

        Args:
            filepath: Path to load agent state from
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.q_table = state["q_table"]
        self.epsilon = state["epsilon"]
        self.state_mins = state["state_mins"]
        self.state_maxs = state["state_maxs"]
        self.training_steps = state["training_steps"]
        self.episode_count = state["episode_count"]

        if "discretization_levels" in state:
            self.discretization_levels = state["discretization_levels"]

        print(f"Agent loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)}")
