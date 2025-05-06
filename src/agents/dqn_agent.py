#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) Agent Implementation

This module implements Deep Q-Learning for reinforcement learning problems,
particularly for traffic signal control. It includes several advanced features:

- Experience Replay: Stores and reuses past experiences to break correlations
- Prioritized Experience Replay: Samples important transitions more frequently
- Double DQN: Reduces overestimation bias by decoupling action selection and evaluation
- Dueling DQN: Separates state value and action advantage estimation
- N-step Learning: Uses multi-step returns for faster learning
- Soft Target Updates: Gradually updates target network for stability

Key references:
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
- "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
- "Prioritized Experience Replay" (Schaul et al., 2016)
"""

from collections import deque
from typing import Tuple, List, Dict, Optional, Union, Any
import os
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:
    """Experience replay buffer with prioritized sampling capability.
    Stores transitions that the agent observes for later reuse during training.

    Implements prioritized experience replay, which samples transitions with
    high expected learning progress more frequently, leading to more efficient learning.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum size of the buffer
            alpha: Priority exponent parameter (0 = uniform sampling, higher values increase prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        # Create experience tuple
        experience = (state, action, reward, next_state, done)

        # Add to buffer
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        # Set max priority for new experience
        self.priorities[self.position] = (
            1.0 if self.size == 1 else self.priorities.max()
        )

        # Update position
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample a batch of experiences with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple of (batch, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size

        # Calculate sampling probabilities
        if self.alpha == 0:
            probs = np.ones(self.size) / self.size
        else:
            probs = self.priorities[: self.size] ** self.alpha
            probs /= probs.sum()

        # Sample experiences
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** -beta
        weights /= weights.max()  # Normalize weights

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of the transitions
            priorities: New priorities
        """
        self.priorities[indices] = priorities + 1e-5  # Add small constant for stability

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size


class DuelingDQN(nn.Module):
    """Dueling DQN architecture that separates state value and action advantage estimation.

    This network architecture helps the agent learn which states are valuable without
    having to learn the effect of each action for each state, leading to better policy evaluation.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        """Initialize the Dueling Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        features = self.feature_layer(state)

        # Calculate state value
        value = self.value_stream(features)

        # Calculate action advantages
        advantage = self.advantage_stream(features)

        # Combine value and advantage to get Q-values
        # Subtract mean advantage to ensure identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DQNNetwork(nn.Module):
    """Deep Q-Network for traffic signal control.
    This network maps states to Q-values for each action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        use_batch_norm: bool = False,
    ):
        """Initialize the Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            use_batch_norm: Whether to use batch normalization instead of layer normalization
        """
        super(DQNNetwork, self).__init__()

        # Choose normalization layer based on parameter
        norm_layer1 = nn.BatchNorm1d if use_batch_norm else nn.LayerNorm
        norm_layer2 = nn.BatchNorm1d if use_batch_norm else nn.LayerNorm
        norm_layer3 = nn.BatchNorm1d if use_batch_norm else nn.LayerNorm

        # Network with normalization as described in the paper
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            norm_layer1(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            norm_layer2(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            norm_layer3(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        # Initialize weights using Kaiming
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        return self.layers(state)


class DQNAgent:
    """Deep Q-Network agent for traffic signal control.

    This agent implements various advanced DQN techniques to improve stability and efficiency:
    - Experience replay with prioritization
    - Double DQN to reduce overestimation bias
    - Dueling network architecture (optional)
    - N-step returns for faster learning
    - Soft target updates (optional)
    - Huber loss for robustness to outliers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        learning_rate: float = 0.0005,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        tau: float = 1.0,  # 1.0 = hard update, <1.0 = soft update
        double_dqn: bool = True,
        dueling_dqn: bool = False,
        n_step: int = 3,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_steps: int = 100000,
        use_huber_loss: bool = False,
        huber_delta: float = 1.0,
        use_batch_norm: bool = False,
        device: str = None,
    ):
        """Initialize the DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Target network update frequency
            tau: Target network update rate (1.0 = hard update, <1.0 = soft update)
            double_dqn: Whether to use Double DQN
            dueling_dqn: Whether to use Dueling DQN architecture
            n_step: Number of steps for n-step returns
            priority_alpha: Priority exponent for prioritized replay
            priority_beta_start: Initial importance sampling exponent
            priority_beta_end: Final importance sampling exponent
            priority_beta_steps: Steps to anneal beta from start to end
            use_huber_loss: Whether to use Huber loss instead of MSE
            huber_delta: Delta parameter for Huber loss
            use_batch_norm: Whether to use batch normalization
            device: Device to use for training (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.n_step = n_step
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Prioritized replay parameters
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta_start
        self.priority_beta_start = priority_beta_start
        self.priority_beta_end = priority_beta_end
        self.priority_beta_steps = priority_beta_steps

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize networks based on architecture choice
        if self.dueling_dqn:
            self.q_network = DuelingDQN(state_dim, action_dim, hidden_size).to(
                self.device
            )
            self.target_network = DuelingDQN(state_dim, action_dim, hidden_size).to(
                self.device
            )
        else:
            self.q_network = DQNNetwork(
                state_dim, action_dim, hidden_size, use_batch_norm
            ).to(self.device)
            self.target_network = DQNNetwork(
                state_dim, action_dim, hidden_size, use_batch_norm
            ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, alpha=priority_alpha)

        # N-step return buffer
        self.n_step_buffer = deque(maxlen=n_step)

        # Training tracking
        self.train_step = 0
        self.update_count = 0

        # Metrics for visualization
        self.loss_history = []
        self.reward_history = []
        self.q_value_history = []
        self.epsilon_history = []

    def act(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            evaluate: Whether to evaluate (no exploration)

        Returns:
            Selected action
        """
        # Pure exploitation during evaluation
        if evaluate:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

        # Exploration
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

            # Store max Q-value for visualization
            if (
                len(self.q_value_history) < 10000
            ):  # Limit storage to prevent memory issues
                self.q_value_history.append(torch.max(q_values).item())

        return torch.argmax(q_values).item()

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store experience and perform learning update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated

        Returns:
            Loss value if learning update was performed, None otherwise
        """
        # Store in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If n-step buffer is full, compute n-step return and store in replay buffer
        if len(self.n_step_buffer) == self.n_step:
            state_0, action_0, _, _, _ = self.n_step_buffer[0]
            _, _, _, next_state_n, done_n = self.n_step_buffer[-1]

            # Calculate n-step discounted return
            n_step_reward = 0
            for i in range(self.n_step):
                n_step_reward += (self.gamma**i) * self.n_step_buffer[i][2]

            # Add to replay buffer
            self.replay_buffer.add(
                state_0, action_0, n_step_reward, next_state_n, done_n
            )

        # Regular update for the last step (if needed)
        if done and len(self.n_step_buffer) < self.n_step:
            # Store all remaining transitions
            while self.n_step_buffer:
                state_0, action_0, _, _, _ = self.n_step_buffer[0]
                _, _, _, next_state_n, done_n = self.n_step_buffer[-1]

                # Calculate multi-step discounted return
                n_step_reward = 0
                for i in range(len(self.n_step_buffer)):
                    n_step_reward += (self.gamma**i) * self.n_step_buffer[i][2]

                # Add to replay buffer
                self.replay_buffer.add(
                    state_0, action_0, n_step_reward, next_state_n, done_n
                )

                # Remove first item
                self.n_step_buffer.popleft()

        # Track reward for visualization
        if len(self.reward_history) < 10000:  # Limit storage to prevent memory issues
            self.reward_history.append(reward)

        # Perform learning update if enough samples are available
        loss = None
        if len(self.replay_buffer) >= self.batch_size:
            loss = self._update_network()

            # Update target network periodically (hard update) or gradually (soft update)
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                if self.tau >= 1.0:  # Hard update
                    self.target_network.load_state_dict(self.q_network.state_dict())
                else:  # Soft update
                    for target_param, param in zip(
                        self.target_network.parameters(), self.q_network.parameters()
                    ):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Track epsilon for visualization
            if (
                len(self.epsilon_history) < 10000
            ):  # Limit storage to prevent memory issues
                self.epsilon_history.append(self.epsilon)

            # Update beta for prioritized replay
            self.priority_beta = min(
                self.priority_beta_end,
                self.priority_beta
                + (self.priority_beta_end - self.priority_beta_start)
                / self.priority_beta_steps,
            )

        return loss

    def _update_network(self) -> float:
        """Perform a gradient update step on the Q-network.

        Returns:
            Loss value from the update
        """
        # Sample from replay buffer
        batch, indices, weights = self.replay_buffer.sample(
            self.batch_size, beta=self.priority_beta
        )

        # Extract batch
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Compute current Q-values
        q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # Compute next Q-values using Double DQN if enabled
        with torch.no_grad():
            if self.double_dqn:
                # Select actions using online network
                next_actions = self.q_network(next_states_tensor).argmax(
                    1, keepdim=True
                )
                # Evaluate Q-values using target network
                next_q_values = self.target_network(next_states_tensor).gather(
                    1, next_actions
                )
            else:
                # Standard DQN: Use target network for both action selection and evaluation
                next_q_values = self.target_network(next_states_tensor).max(
                    1, keepdim=True
                )[0]

            # Compute target Q-values
            target_q_values = (
                rewards_tensor
                + (1 - dones_tensor) * (self.gamma**self.n_step) * next_q_values
            )

        # Compute TD errors for prioritized replay update
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors.squeeze())

        # Compute loss (weighted MSE or Huber)
        if self.use_huber_loss:
            # Huber loss is more robust to outliers
            element_wise_loss = F.smooth_l1_loss(
                q_values, target_q_values, reduction="none", beta=self.huber_delta
            )
        else:
            # Mean squared error
            element_wise_loss = F.mse_loss(q_values, target_q_values, reduction="none")

        # Apply importance sampling weights
        loss = (weights_tensor * element_wise_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        self.train_step += 1

        # Track loss for visualization
        loss_value = loss.item()
        if len(self.loss_history) < 10000:  # Limit storage to prevent memory issues
            self.loss_history.append(loss_value)

        return loss_value

    def save(self, filepath: str):
        """Save the agent's model and parameters.

        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self.train_step,
                "update_count": self.update_count,
                "priority_beta": self.priority_beta,
            },
            filepath,
        )

        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the agent's model and parameters.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.train_step = checkpoint["train_step"]
        self.update_count = checkpoint["update_count"]
        if "priority_beta" in checkpoint:
            self.priority_beta = checkpoint["priority_beta"]

        print(f"Model loaded from {filepath}")

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics for visualization.

        Args:
            save_path: Path to save the plot images (if None, plots are displayed)
        """
        if (
            not self.loss_history
            and not self.reward_history
            and not self.q_value_history
        ):
            print("No metrics to plot yet.")
            return

        # Create directory if save_path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Set up the figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss history
        if self.loss_history:
            axs[0, 0].plot(self.loss_history)
            axs[0, 0].set_title("Loss History")
            axs[0, 0].set_xlabel("Training Steps")
            axs[0, 0].set_ylabel("Loss")

        # Plot reward history
        if self.reward_history:
            # Plot rolling average for smoother visualization
            window_size = min(100, len(self.reward_history))
            rolling_avg = np.convolve(
                self.reward_history, np.ones(window_size) / window_size, mode="valid"
            )
            axs[0, 1].plot(rolling_avg)
            axs[0, 1].set_title(f"Reward History (Rolling Avg: {window_size})")
            axs[0, 1].set_xlabel("Steps")
            axs[0, 1].set_ylabel("Reward")

        # Plot Q-value history
        if self.q_value_history:
            # Plot rolling average for smoother visualization
            window_size = min(100, len(self.q_value_history))
            rolling_avg = np.convolve(
                self.q_value_history, np.ones(window_size) / window_size, mode="valid"
            )
            axs[1, 0].plot(rolling_avg)
            axs[1, 0].set_title(f"Q-Value History (Rolling Avg: {window_size})")
            axs[1, 0].set_xlabel("Steps")
            axs[1, 0].set_ylabel("Max Q-Value")

        # Plot epsilon history
        if self.epsilon_history:
            axs[1, 1].plot(self.epsilon_history)
            axs[1, 1].set_title("Exploration Rate (Epsilon)")
            axs[1, 1].set_xlabel("Training Steps")
            axs[1, 1].set_ylabel("Epsilon")

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, "dqn_metrics.png"))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def get_hyperparameter_presets(preset: str = "default") -> Dict[str, Any]:
        """Get predefined hyperparameter presets for different scenarios.

        Args:
            preset: Name of the preset to use ('default', 'traffic', 'stable', 'fast')

        Returns:
            Dictionary of hyperparameters
        """
        presets = {
            "default": {
                "hidden_size": 256,
                "learning_rate": 0.0005,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "buffer_size": 10000,
                "batch_size": 64,
                "target_update_freq": 10,
                "tau": 1.0,
                "double_dqn": True,
                "dueling_dqn": False,
                "n_step": 3,
                "priority_alpha": 0.6,
                "priority_beta_start": 0.4,
                "use_huber_loss": False,
            },
            "traffic": {  # Optimized for traffic signal control
                "hidden_size": 256,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.997,
                "buffer_size": 50000,
                "batch_size": 128,
                "target_update_freq": 20,
                "tau": 0.005,  # Soft updates
                "double_dqn": True,
                "dueling_dqn": True,
                "n_step": 5,
                "priority_alpha": 0.6,
                "priority_beta_start": 0.4,
                "use_huber_loss": True,
            },
            "stable": {  # Prioritizes stability over speed
                "hidden_size": 256,
                "learning_rate": 0.0001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.999,
                "buffer_size": 100000,
                "batch_size": 64,
                "target_update_freq": 50,
                "tau": 0.001,  # Very soft updates
                "double_dqn": True,
                "dueling_dqn": True,
                "n_step": 1,  # Standard 1-step returns
                "priority_alpha": 0.5,
                "priority_beta_start": 0.4,
                "use_huber_loss": True,
            },
            "fast": {  # Prioritizes fast learning
                "hidden_size": 128,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 0.99,
                "buffer_size": 10000,
                "batch_size": 32,
                "target_update_freq": 5,
                "tau": 1.0,  # Hard updates
                "double_dqn": False,  # Simpler standard DQN
                "dueling_dqn": False,
                "n_step": 3,
                "priority_alpha": 0.7,  # Higher prioritization
                "priority_beta_start": 0.5,
                "use_huber_loss": True,
            },
        }

        if preset not in presets:
            print(f"Preset '{preset}' not found, using 'default' instead.")
            preset = "default"

        return presets[preset]


# Example usage
if __name__ == "__main__":
    # Create a simple environment for testing
    import gymnasium as gym

    # Create a simple CartPole environment for testing
    env = gym.make("CartPole-v1")

    # Create DQN agent with traffic preset
    hyperparams = DQNAgent.get_hyperparameter_presets("traffic")

    # Adjust for the simple environment
    hyperparams["buffer_size"] = 5000
    hyperparams["batch_size"] = 32

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        **hyperparams,
    )

    # Train for a few episodes
    num_episodes = 5
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            # Select action
            action = agent.act(state)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Learn
            agent.learn(state, action, reward, next_state, done)

            # Update state
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: reward = {episode_reward}")

    print(
        f"Average reward over {num_episodes} episodes: {sum(total_rewards)/num_episodes}"
    )

    # Plot training metrics
    agent.plot_metrics()
