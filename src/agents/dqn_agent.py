#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deep Q-Network (DQN) Agent"""

from collections import deque
from typing import Tuple, List

import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:
    """Experience replay buffer with prioritized sampling capability.
    Stores transitions that the agent observes for later reuse during training."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum size of the buffer
            alpha: Priority exponent parameter (0 = uniform sampling)
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
            beta: Importance sampling exponent (0 = no correction)

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


class DQNNetwork(nn.Module):
    """Deep Q-Network for traffic signal control.
    This network maps states to Q-values for each action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        """Initialize the Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()

        # Network with layer normalization as described in the paper
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        # Initialize weights using Kaiming
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
        return self.layers(state)


class DQNAgent:
    """Deep Q-Network agent for traffic signal control."""

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
        double_dqn: bool = True,
        n_step: int = 3,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_steps: int = 100000,
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
            double_dqn: Whether to use Double DQN
            n_step: Number of steps for n-step returns
            priority_alpha: Priority exponent for prioritized replay
            priority_beta_start: Initial importance sampling exponent
            priority_beta_end: Final importance sampling exponent
            priority_beta_steps: Steps to anneal beta from start to end
            device: Device to use for training (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.n_step = n_step

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

        # Initialize networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_size).to(
            self.device
        )
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

    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Exploration
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return torch.argmax(q_values).item()

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience and perform learning update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
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

        # Perform learning update if enough samples are available
        if len(self.replay_buffer) >= self.batch_size:
            self._update_network()

            # Update target network periodically
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Update beta for prioritized replay
            self.priority_beta = min(
                self.priority_beta_end,
                self.priority_beta
                + (self.priority_beta_end - self.priority_beta_start)
                / self.priority_beta_steps,
            )

    def _update_network(self):
        """Perform a gradient update step on the Q-network."""
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

        # Compute TD errors
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors.squeeze())

        # Compute loss (weighted MSE)
        loss = (
            weights_tensor * F.mse_loss(q_values, target_q_values, reduction="none")
        ).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        self.train_step += 1

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

        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Create a simple environment for testing
    import gymnasium as gym

    # Create a simple CartPole environment for testing
    env = gym.make("CartPole-v1")

    # Create DQN agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.n
    )

    # Test agent
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        # Select action
        action = agent.act(state)

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)

        # Learn
        agent.learn(state, action, reward, next_state, done)

        # Update state
        state = next_state
        total_reward += reward

    print(f"Episode reward: {total_reward}")
