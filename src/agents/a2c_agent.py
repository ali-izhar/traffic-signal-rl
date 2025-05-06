#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advantage Actor-Critic (A2C) Agent Implementation

This module implements the A2C algorithm, which combines policy-based and value-based methods
for reinforcement learning. The algorithm uses a policy network (actor) that determines which
action to take, and a value network (critic) that evaluates the expected return from each state.

Key components:
- SharedNetwork: Common feature extraction layers for both actor and critic
- ActorNetwork: Policy network that outputs action probabilities
- CriticNetwork: Value network that estimates state values
- A2CAgent: Main agent class that handles training and action selection

The agent uses Generalized Advantage Estimation (GAE) to calculate advantages,
which reduces variance while maintaining an acceptable level of bias.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


class SharedNetwork(nn.Module):
    """
    Shared feature extraction layers for both actor and critic networks.

    This network extracts relevant features from the state representation,
    which are then fed into separate actor and critic heads.

    Args:
        state_dim (int): Dimension of the state space
        hidden_sizes (list): List of hidden layer sizes
    """

    def __init__(self, state_dim, hidden_sizes=[256, 128]):
        super(SharedNetwork, self).__init__()

        # Feature extraction layers
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
        )

        # Initialize weights using Kaiming initialization
        for layer in self.feature_extraction:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        """Extract features from state"""
        return self.feature_extraction(state)


class ActorNetwork(nn.Module):
    """
    Policy network (Actor) for discrete action spaces.

    This network outputs a probability distribution over actions,
    from which actions are sampled during training.

    Args:
        feature_dim (int): Dimension of the feature representation
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of the hidden layer
    """

    def __init__(self, feature_dim, action_dim, hidden_size=64):
        super(ActorNetwork, self).__init__()

        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.LayerNorm(action_dim),
        )

        # Initialize weights with orthogonal initialization (helps with training stability)
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Compute action logits"""
        return self.actor_head(features)

    def get_action_and_log_prob(self, features, deterministic=False):
        """
        Get action and log probability

        Args:
            features (torch.Tensor): Features extracted from the state
            deterministic (bool): If True, select the action with highest probability
                                  If False, sample from the action distribution

        Returns:
            action (torch.Tensor): Selected action
            log_prob (torch.Tensor): Log probability of the selected action
            entropy (torch.Tensor): Entropy of the action distribution
        """
        logits = self.forward(features)

        # Add small epsilon to prevent numerical instability
        logits = logits.clamp(-10.0, 10.0)

        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value network (Critic) that estimates the value function V(s).

    This network estimates the expected return from a given state,
    which is used to compute advantages for the actor update.

    Args:
        feature_dim (int): Dimension of the feature representation
        hidden_size (int): Size of the hidden layer
    """

    def __init__(self, feature_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()

        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize weights with orthogonal initialization
        for layer in self.critic_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Compute state value"""
        return self.critic_head(features)


class A2CAgent:
    """
    Advantage Actor-Critic agent with entropy regularization.

    This agent implements the A2C algorithm, which combines policy-based and value-based
    methods for reinforcement learning. It uses a policy network (actor) that determines
    which action to take, and a value network (critic) that evaluates the expected return
    from each state.

    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        device (str): Device to run the model on ('cuda' or 'cpu')
        lr (float): Learning rate for the optimizer
        gamma (float): Discount factor for future rewards
        gae_lambda (float): Lambda parameter for GAE
        entropy_coef (float): Coefficient for entropy regularization
        value_coef (float): Coefficient for value loss
        max_grad_norm (float): Maximum norm for gradient clipping
        hidden_sizes (list): List of hidden layer sizes for networks
        checkpoint_dir (str, optional): Directory for model checkpoints. Not used directly
                                    for saving models, but kept for backward compatibility.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=5e-4,
        gamma=0.95,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        hidden_sizes=[256, 128, 64],
        checkpoint_dir=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir

        # Create networks
        self.shared_network = SharedNetwork(state_dim, hidden_sizes[:2]).to(device)
        self.actor = ActorNetwork(hidden_sizes[1], action_dim, hidden_sizes[2]).to(
            device
        )
        self.critic = CriticNetwork(hidden_sizes[1], hidden_sizes[2]).to(device)

        # Create optimizer with combined parameters from all networks
        params = (
            list(self.shared_network.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )

        self.optimizer = optim.Adam(params, lr=lr, weight_decay=0.01)

        # Setup learning rate scheduler with warm-up and cosine decay
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=1e-5
        )

        # Ensure checkpoint directory exists if provided
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # For tracking training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "entropies": [],
        }

    def act(self, state, deterministic=False):
        """
        Select action based on current policy.

        Args:
            state (np.ndarray or torch.Tensor): Current state observation
            deterministic (bool): If True, select the action with highest probability
                                 If False, sample from the action distribution

        Returns:
            int: Selected action
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            features = self.shared_network(state)
            action, _, _ = self.actor.get_action_and_log_prob(features, deterministic)

        return action.cpu().item()

    def compute_gae(self, rewards, values, next_value, dones, gamma, gae_lambda):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE is a method to estimate advantages that balances bias vs. variance
        by combining multi-step returns at different time scales.

        Args:
            rewards (list): List of rewards
            values (list): List of state values
            next_value (float): Value of the next state
            dones (list): List of done flags
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter

        Returns:
            torch.Tensor: Tensor of advantage estimates
        """
        # Create tensors for advantages and returns
        advantages = []
        gae = 0

        # Compute GAE going backwards from the end of the trajectory
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]

            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[i] + gamma * next_val * (1 - dones[i]) - values[i]

            # Advantage: sum of discounted TD errors
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update policy and value networks.

        Args:
            states (list): List of states
            actions (list): List of actions
            rewards (list): List of rewards
            next_states (list): List of next states
            dones (list): List of done flags

        Returns:
            dict: Dictionary of training metrics
        """
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # --- 1. Extract features from states ---
        features = self.shared_network(states)
        next_features = self.shared_network(next_states)

        # --- 2. Get values from critic ---
        values = self.critic(features).squeeze(-1)
        next_values = self.critic(next_features).squeeze(-1)

        # --- 3. Compute advantages using GAE ---
        advantages = self.compute_gae(
            rewards.cpu().numpy(),
            values.detach().cpu().numpy(),
            next_values[-1].detach().cpu().numpy(),
            dones.cpu().numpy(),
            self.gamma,
            self.gae_lambda,
        )

        # --- 4. Normalize advantages (reduces variance) ---
        if len(advantages) > 1:  # Only normalize if we have more than one sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 5. Get log probabilities and entropy from actor ---
        _, log_probs, entropy = self.actor.get_action_and_log_prob(features, False)

        # --- 6. Compute returns for value loss ---
        returns = advantages + values.detach()

        # --- 7. Calculate losses ---
        # Actor loss: -log_prob * advantage (negative because we're doing gradient descent)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss: mean squared error between predicted values and returns
        critic_loss = F.mse_loss(values, returns)

        # Total loss with entropy regularization
        # Entropy encourages exploration by penalizing deterministic policies
        total_loss = (
            actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        )

        # --- 8. Perform optimization step ---
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.shared_network.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()

        # Store metrics for monitoring
        metrics = {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

        # Update training metrics
        self.training_metrics["actor_losses"].append(actor_loss.item())
        self.training_metrics["critic_losses"].append(critic_loss.item())
        self.training_metrics["entropies"].append(entropy.item())

        return metrics

    def record_episode_metrics(self, episode_reward, episode_length):
        """
        Record metrics from a completed episode.

        Args:
            episode_reward (float): Total reward for the episode
            episode_length (int): Total number of steps in the episode
        """
        self.training_metrics["episode_rewards"].append(episode_reward)
        self.training_metrics["episode_lengths"].append(episode_length)

    def plot_training_metrics(self, smooth_window=10):
        """
        Plot training metrics.

        Args:
            smooth_window (int): Window size for smoothing metrics
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot episode rewards
        rewards = self.training_metrics["episode_rewards"]
        if len(rewards) > 0:
            axs[0, 0].plot(rewards, alpha=0.6, label="Raw")
            if len(rewards) >= smooth_window:
                smooth_rewards = np.convolve(
                    rewards, np.ones(smooth_window) / smooth_window, mode="valid"
                )
                axs[0, 0].plot(
                    range(smooth_window - 1, smooth_window - 1 + len(smooth_rewards)),
                    smooth_rewards,
                    label="Smoothed",
                )
            axs[0, 0].set_title("Episode Rewards")
            axs[0, 0].set_xlabel("Episode")
            axs[0, 0].set_ylabel("Total Reward")
            axs[0, 0].legend()

        # Plot episode lengths
        lengths = self.training_metrics["episode_lengths"]
        if len(lengths) > 0:
            axs[0, 1].plot(lengths, alpha=0.6, label="Raw")
            if len(lengths) >= smooth_window:
                smooth_lengths = np.convolve(
                    lengths, np.ones(smooth_window) / smooth_window, mode="valid"
                )
                axs[0, 1].plot(
                    range(smooth_window - 1, smooth_window - 1 + len(smooth_lengths)),
                    smooth_lengths,
                    label="Smoothed",
                )
            axs[0, 1].set_title("Episode Lengths")
            axs[0, 1].set_xlabel("Episode")
            axs[0, 1].set_ylabel("Steps")
            axs[0, 1].legend()

        # Plot losses
        actor_losses = self.training_metrics["actor_losses"]
        critic_losses = self.training_metrics["critic_losses"]
        if len(actor_losses) > 0 and len(critic_losses) > 0:
            axs[1, 0].plot(actor_losses, label="Actor Loss")
            axs[1, 0].plot(critic_losses, label="Critic Loss")
            axs[1, 0].set_title("Losses")
            axs[1, 0].set_xlabel("Update")
            axs[1, 0].set_ylabel("Loss")
            axs[1, 0].legend()

        # Plot entropy
        entropies = self.training_metrics["entropies"]
        if len(entropies) > 0:
            axs[1, 1].plot(entropies)
            axs[1, 1].set_title("Policy Entropy")
            axs[1, 1].set_xlabel("Update")
            axs[1, 1].set_ylabel("Entropy")

        plt.tight_layout()
        plt.show()

    def collect_trajectory(self, env, max_steps=1000):
        """
        Collect a trajectory (sequence of experience tuples) by interacting with the environment.

        Args:
            env: Environment to interact with
            max_steps (int): Maximum number of steps to take

        Returns:
            dict: Dictionary containing lists of states, actions, rewards, next_states, and dones
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward
            steps += 1

        # Record episode metrics
        self.record_episode_metrics(total_reward, steps)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def save(self, filepath):
        """
        Save agent state.

        Args:
            filepath (str): Full path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(
            {
                "shared_network": self.shared_network.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "training_metrics": self.training_metrics,
            },
            filepath,
        )

    def load(self, filepath):
        """
        Load agent state.

        Args:
            filepath (str): Full path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.shared_network.load_state_dict(checkpoint["shared_network"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "training_metrics" in checkpoint:
            self.training_metrics = checkpoint["training_metrics"]
