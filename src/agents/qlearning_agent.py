#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tabular Q-Learning Agent for Traffic Signal Control

Q-learning is a model-free reinforcement learning algorithm that learns to make optimal
decisions by estimating the value of taking actions in different states through
experience. It uses a tabular approach to store and update Q-values, which represent
the expected future rewards for state-action pairs.

Key components:
- Discretization of continuous state space into buckets for tabular representation
- Epsilon-greedy exploration strategy with adaptive annealing
- Temporal Difference (TD) learning for Q-value updates
- Experience replay for more stable learning (optional)

Learning update rule:
    Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
    where:
    - α is the learning rate
    - γ is the discount factor
    - s' is the next state
    - R is the immediate reward
    - max(Q(s',a')) is the maximum Q-value for the next state
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union, Optional, Any
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for more stable learning.

    Stores experiences and allows random sampling to break
    correlations between consecutive samples.
    """

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class QLearningAgent:
    """Tabular Q-learning agent for traffic signal control.

    This agent discretizes the continuous state space and uses a table to store
    Q-values for state-action pairs. It implements standard Q-learning with
    optional enhancements like experience replay and adaptive exploration.
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
        use_double_q: bool = False,
        use_experience_replay: bool = False,
        replay_buffer_size: int = 10000,
        replay_batch_size: int = 32,
        checkpoint_dir: str = None,
    ):
        """Initialize the Q-Learning agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate (alpha) for updating Q-values
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            discretization_levels: Number of bins for discretizing continuous states
            use_double_q: Whether to use Double Q-learning algorithm
            use_experience_replay: Whether to use experience replay
            replay_buffer_size: Size of replay buffer if using experience replay
            replay_batch_size: Batch size for experience replay updates
            checkpoint_dir: Optional directory for model checkpoints
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

        # Advanced learning parameters
        self.use_double_q = use_double_q
        self.use_experience_replay = use_experience_replay
        self.replay_batch_size = replay_batch_size

        # Initialize experience replay if enabled
        if self.use_experience_replay:
            self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

        # Initialize secondary Q-table for double Q-learning
        if self.use_double_q:
            self.q_table_secondary = {}

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
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Training metrics
        self.training_steps = 0
        self.episode_count = 0
        self.rewards_history = []
        self.td_errors_history = []
        self.q_values_history = []

    def discretize_state(self, state: np.ndarray) -> Tuple:
        """Discretize a continuous state into a discrete state usable as Q-table key.

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

    def get_q_values(self, discrete_state):
        """Get Q-values for a discrete state, initializing if necessary.

        Args:
            discrete_state: Discretized state tuple

        Returns:
            Array of Q-values for each action
        """
        if discrete_state not in self.q_table:
            # Initialize with small random values if state not seen before
            self.q_table[discrete_state] = np.random.uniform(
                low=0, high=0.1, size=self.action_dim
            )

            # Also initialize the secondary Q-table if using double Q-learning
            if self.use_double_q and discrete_state not in self.q_table_secondary:
                self.q_table_secondary[discrete_state] = np.random.uniform(
                    low=0, high=0.1, size=self.action_dim
                )

        return self.q_table[discrete_state]

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            eval_mode: If True, use greedy policy (no exploration)

        Returns:
            Selected action
        """
        # Discretize state
        discrete_state = self.discretize_state(state)

        # Exploration (only if not in evaluation mode)
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # Exploitation - get Q-values, initializing if necessary
        q_values = self.get_q_values(discrete_state)
        return np.argmax(q_values)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """Update Q-value based on observed transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done

        Returns:
            Dictionary with learning metrics
        """
        # Add to experience replay buffer if enabled
        if self.use_experience_replay:
            self.replay_buffer.add(state, action, reward, next_state, done)

        # Perform standard Q-learning update
        metrics = self._update_q_values(state, action, reward, next_state, done)

        # Perform experience replay if enabled and buffer has enough samples
        if (
            self.use_experience_replay
            and len(self.replay_buffer) >= self.replay_batch_size
        ):
            self._perform_experience_replay()

        # Update training statistics
        self.training_steps += 1
        self.rewards_history.append(reward)
        self.td_errors_history.append(metrics["td_error"])
        self.q_values_history.append(metrics["q_value"])

        if done:
            self.episode_count += 1

        # Decay exploration rate
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start * np.exp(-self.epsilon_decay * self.episode_count),
        )

        # Add epsilon to metrics
        metrics["epsilon"] = self.epsilon
        metrics["q_table_size"] = len(self.q_table)

        return metrics

    def _update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """Update Q-values based on transition.

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

        # Get current Q-value
        q_values = self.get_q_values(discrete_state)
        current_q = q_values[action]

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            # Get next state Q-values
            next_q_values = self.get_q_values(discrete_next_state)

            if self.use_double_q:
                # Double Q-learning approach
                # Use primary Q-table for action selection, secondary for evaluation
                best_action = np.argmax(next_q_values)

                # Ensure secondary Q-table is initialized for next state
                if discrete_next_state not in self.q_table_secondary:
                    self.q_table_secondary[discrete_next_state] = np.random.uniform(
                        low=0, high=0.1, size=self.action_dim
                    )

                target_q = (
                    reward
                    + self.gamma
                    * self.q_table_secondary[discrete_next_state][best_action]
                )

                # Also update secondary Q-table (with 50% probability to maintain independence)
                if np.random.random() < 0.5:
                    # Swap primary and secondary roles
                    current_q_secondary = self.q_table_secondary[discrete_state][action]
                    best_action_secondary = np.argmax(
                        self.q_table_secondary[discrete_next_state]
                    )
                    target_q_secondary = (
                        reward + self.gamma * next_q_values[best_action_secondary]
                    )
                    td_error_secondary = target_q_secondary - current_q_secondary
                    self.q_table_secondary[discrete_state][action] += (
                        self.lr * td_error_secondary
                    )
            else:
                # Standard Q-learning approach
                target_q = reward + self.gamma * np.max(next_q_values)

        # Update Q-value
        td_error = target_q - current_q
        q_values[action] += self.lr * td_error

        return {
            "td_error": td_error,
            "q_value": current_q,
        }

    def _perform_experience_replay(self) -> None:
        """Perform experience replay updates."""
        # Sample a batch of experiences
        experiences = self.replay_buffer.sample(self.replay_batch_size)

        # Update Q-values for each experience
        for state, action, reward, next_state, done in experiences:
            self._update_q_values(state, action, reward, next_state, done)

    def save(self, filepath: str) -> None:
        """Save Q-table and agent parameters.

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
            "rewards_history": self.rewards_history,
            "td_errors_history": self.td_errors_history,
            "q_values_history": self.q_values_history,
        }

        if self.use_double_q:
            state["q_table_secondary"] = self.q_table_secondary

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"Agent saved to {filepath}")
        print(f"Q-table size: {len(self.q_table)}")

    def load(self, filepath: str) -> None:
        """Load Q-table and agent parameters.

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

        if "q_table_secondary" in state and self.use_double_q:
            self.q_table_secondary = state["q_table_secondary"]

        # Load histories if available
        self.rewards_history = state.get("rewards_history", [])
        self.td_errors_history = state.get("td_errors_history", [])
        self.q_values_history = state.get("q_values_history", [])

        print(f"Agent loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)}")

    def plot_learning_progress(
        self,
        moving_avg_window: int = 100,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot learning progress metrics.

        Args:
            moving_avg_window: Window size for moving average calculation
            figsize: Size of the figure
            save_path: Path to save the figure. If None, figure is displayed interactively.
        """
        if not self.rewards_history:
            print("No training data available to plot.")
            return

        rewards = self.rewards_history
        td_errors = self.td_errors_history
        q_values = self.q_values_history

        # Calculate moving averages
        def moving_average(data, window):
            return np.convolve(data, np.ones(window) / window, mode="valid")

        if len(rewards) >= moving_avg_window:
            rewards_ma = moving_average(rewards, moving_avg_window)
            td_errors_ma = moving_average(td_errors, moving_avg_window)
            q_values_ma = moving_average(q_values, moving_avg_window)
            x_ma = range(moving_avg_window - 1, moving_avg_window - 1 + len(rewards_ma))
        else:
            rewards_ma, td_errors_ma, q_values_ma, x_ma = [], [], [], []

        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Plot rewards
        axes[0].plot(rewards, alpha=0.3, color="blue", label="Rewards")
        if len(rewards_ma) > 0:
            axes[0].plot(
                x_ma,
                rewards_ma,
                color="blue",
                linewidth=2,
                label=f"Moving Avg ({moving_avg_window})",
            )
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Rewards")
        axes[0].legend()
        axes[0].grid(True)

        # Plot TD errors
        axes[1].plot(td_errors, alpha=0.3, color="red", label="TD Errors")
        if len(td_errors_ma) > 0:
            axes[1].plot(
                x_ma,
                td_errors_ma,
                color="red",
                linewidth=2,
                label=f"Moving Avg ({moving_avg_window})",
            )
        axes[1].set_ylabel("TD Error")
        axes[1].set_title("TD Errors")
        axes[1].legend()
        axes[1].grid(True)

        # Plot Q-values
        axes[2].plot(q_values, alpha=0.3, color="green", label="Q-values")
        if len(q_values_ma) > 0:
            axes[2].plot(
                x_ma,
                q_values_ma,
                color="green",
                linewidth=2,
                label=f"Moving Avg ({moving_avg_window})",
            )
        axes[2].set_xlabel("Training Steps")
        axes[2].set_ylabel("Q-value")
        axes[2].set_title("Q-values Evolution")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_q_value_heatmap(
        self,
        max_states: int = 50,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot heatmap of Q-values for visualization.

        Args:
            max_states: Maximum number of states to display
            figsize: Size of the figure
            save_path: Path to save the figure. If None, figure is displayed interactively.
        """
        if not self.q_table:
            print("Q-table is empty. No data to visualize.")
            return

        # Select a subset of states to visualize
        states = list(self.q_table.keys())
        if len(states) > max_states:
            states = random.sample(states, max_states)

        # Create data for heatmap
        q_values_array = np.array([self.q_table[s] for s in states])

        # Create plot
        plt.figure(figsize=figsize)
        plt.imshow(q_values_array, cmap="viridis", aspect="auto")
        plt.colorbar(label="Q-value")
        plt.xlabel("Actions")
        plt.ylabel("States (sample)")
        plt.title("Q-values Heatmap")

        # Add action indices
        plt.xticks(range(self.action_dim))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_state_coverage(self) -> Dict[str, Any]:
        """Calculate statistics about state space coverage.

        Returns:
            Dictionary with coverage statistics
        """
        if not self.q_table:
            return {"covered_states": 0, "coverage_metrics": "No data available"}

        # Calculate coverage statistics
        states = list(self.q_table.keys())
        covered_states = len(states)

        # Analyze state distribution per dimension
        dimension_values = [[] for _ in range(self.state_dim)]
        for state in states:
            for i, value in enumerate(state):
                dimension_values[i].append(value)

        # Calculate statistics for each dimension
        dimension_stats = []
        for i, values in enumerate(dimension_values):
            unique_values = len(set(values))
            possible_values = self.discretization_levels[i]
            coverage_percent = (
                (unique_values / possible_values) * 100 if possible_values > 0 else 0
            )
            dimension_stats.append(
                {
                    "dimension": i,
                    "unique_values": unique_values,
                    "possible_values": possible_values,
                    "coverage_percent": coverage_percent,
                }
            )

        return {"covered_states": covered_states, "dimension_stats": dimension_stats}
