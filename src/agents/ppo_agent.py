#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proximal Policy Optimization (PPO) Agent Implementation

PPO is a state-of-the-art policy gradient algorithm that uses a clipped surrogate objective
to ensure stable policy updates. It combines the sample efficiency of trust-region methods
with the simplicity of implementation of vanilla policy gradient methods.

Key components:
- Actor-Critic Architecture: Uses policy (actor) and value (critic) networks
- Clipped Surrogate Objective: Prevents too large policy updates
- Generalized Advantage Estimation (GAE): Reduces variance in advantage estimates
- Early Stopping: Based on KL divergence between old and new policies
- Mini-batch Updates: Improves sample efficiency by updating multiple times on the same data

References:
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
  High-dimensional continuous control using generalized advantage estimation.
  arXiv preprint arXiv:1506.02438.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt


class SharedNetwork(nn.Module):
    """
    Shared feature extraction layers for both actor and critic networks.

    This architecture enables shared representation learning between policy and value functions,
    improving sample efficiency and stability.
    """

    def __init__(
        self,
        state_dim,
        hidden_sizes=[256, 128],
        activation=nn.ReLU,
        use_batch_norm=False,
        use_layer_norm=True,
    ):
        super(SharedNetwork, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        layers = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(hidden_sizes[0]))
        layers.append(activation())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_sizes[i + 1]))
            layers.append(activation())

        self.feature_extraction = nn.Sequential(*layers)

        # Initialize weights using orthogonal initialization for better performance
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, state):
        """Extract features from state"""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_extraction(state)
        return features


class ActorNetwork(nn.Module):
    """
    Policy network (Actor) for discrete action spaces.

    Maps state features to a probability distribution over actions.
    In discrete action spaces, this is typically a categorical distribution.
    """

    def __init__(self, feature_dim, action_dim, hidden_size=64, activation=nn.ReLU):
        super(ActorNetwork, self).__init__()

        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            activation(),
            nn.Linear(hidden_size, action_dim),
        )

        # Initialize weights using orthogonal initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, features):
        """Compute action logits"""
        logits = self.actor_head(features)
        return logits

    def get_distribution(self, features):
        """
        Get action distribution.

        For discrete action spaces, we use a Categorical distribution over action logits.
        """
        logits = self.forward(features)
        return Categorical(logits=logits)

    def evaluate_actions(self, features, actions):
        """
        Evaluate actions and return log probabilities and entropy.

        Used during training to compute policy loss and entropy bonus.
        """
        distribution = self.get_distribution(features)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy

    def get_action(self, features, deterministic=False):
        """
        Get action, log probability, and entropy.

        Args:
            features: Extracted state features
            deterministic: If True, return the most likely action (argmax);
                          otherwise sample from the distribution

        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            entropy: Entropy of the action distribution
        """
        distribution = self.get_distribution(features)

        if deterministic:
            action = torch.argmax(distribution.probs, dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy


class ContinuousActorNetwork(nn.Module):
    """
    Policy network (Actor) for continuous action spaces.

    Maps state features to a multivariate normal distribution over actions.
    The distribution has a diagonal covariance matrix parameterized by log_std.
    """

    def __init__(
        self,
        feature_dim,
        action_dim,
        hidden_size=64,
        activation=nn.ReLU,
        log_std_init=-0.5,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(ContinuousActorNetwork, self).__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.shared = nn.Sequential(nn.Linear(feature_dim, hidden_size), activation())

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, features):
        """
        Compute action distribution parameters (mean and log_std).

        Returns:
            mean: Mean of the normal distribution
            log_std: Log standard deviation, clamped for numerical stability
        """
        x = self.shared(features)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_distribution(self, features):
        """
        Get action distribution (multivariate normal with diagonal covariance).
        """
        mean, log_std = self.forward(features)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def evaluate_actions(self, features, actions):
        """
        Evaluate actions and return log probabilities and entropy.

        For continuous actions, we compute the log probability density and entropy
        of a multivariate normal distribution.
        """
        mean, log_std = self.forward(features)
        std = torch.exp(log_std)

        # Create multivariate normal distribution with diagonal covariance
        dist = Normal(mean, std)

        # Sum log probs across action dimensions
        log_probs = dist.log_prob(actions).sum(-1)

        # Compute entropy (sum across action dimensions)
        entropy = dist.entropy().sum(-1)

        return log_probs, entropy

    def get_action(self, features, deterministic=False):
        """
        Get action, log probability, and entropy.

        Args:
            features: Extracted state features
            deterministic: If True, return the mean action;
                          otherwise sample from the distribution
        """
        mean, log_std = self.forward(features)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        # Sum log probs and entropy across action dimensions
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value network (Critic) that estimates the state value function V(s).

    The critic's role is to estimate the expected return from a state,
    which is used for advantage estimation in the actor's policy update.
    """

    def __init__(self, feature_dim, hidden_size=64, activation=nn.ReLU):
        super(CriticNetwork, self).__init__()

        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), activation(), nn.Linear(hidden_size, 1)
        )

        # Initialize weights using orthogonal initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    def forward(self, features):
        """Compute state value"""
        return self.critic_head(features).squeeze(-1)


class PPOAgent:
    """
    Proximal Policy Optimization agent implementation.

    PPO is an on-policy actor-critic algorithm that uses a clipped surrogate objective
    to prevent too large policy updates, ensuring more stable learning compared to
    standard policy gradient methods.

    Key features:
    - Clipped surrogate objective to constrain policy updates
    - Generalized Advantage Estimation for more stable advantage estimates
    - Multiple epochs of stochastic gradient descent on the same batch of data
    - Early stopping based on KL divergence to prevent policy divergence
    - Support for both discrete and continuous action spaces
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        continuous_actions=False,
        action_bounds=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        mini_batch_size=64,
        target_kl=0.015,
        hidden_sizes=[256, 128, 64],
        use_batch_norm=False,
        use_layer_norm=True,
        checkpoint_dir=None,
    ):
        """
        Initialize the PPO agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            continuous_actions: Whether the action space is continuous
            action_bounds: Bounds for continuous actions as [min, max]
            device: Device to run the agent on ('cpu' or 'cuda')
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (0 <= lambda <= 1)
            clip_range: PPO clip range (epsilon in the paper)
            critic_coef: Coefficient for value loss term
            entropy_coef: Coefficient for entropy bonus term
            max_grad_norm: Maximum norm for gradient clipping
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Mini-batch size for updates
            target_kl: Target KL divergence threshold for early stopping
            hidden_sizes: Hidden layer sizes for networks
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            checkpoint_dir (str, optional): Directory for model checkpoints. Not used directly
                                       for saving models, but kept for backward compatibility.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        self.action_bounds = action_bounds
        self.device = device

        # PPO hyperparameters
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE lambda parameter
        self.clip_range = clip_range  # PPO clip range
        self.critic_coef = critic_coef  # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy bonus coefficient
        self.max_grad_norm = max_grad_norm  # Gradient clipping threshold
        self.ppo_epochs = ppo_epochs  # Number of PPO update epochs
        self.mini_batch_size = mini_batch_size  # Mini-batch size
        self.target_kl = target_kl  # KL divergence threshold for early stopping
        self.checkpoint_dir = checkpoint_dir  # Directory to save checkpoints

        # Create networks
        self.shared_network = SharedNetwork(
            state_dim,
            hidden_sizes[:2],
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        ).to(device)

        # Create actor network based on action space type
        if continuous_actions:
            self.actor = ContinuousActorNetwork(
                hidden_sizes[1], action_dim, hidden_sizes[2]
            ).to(device)
        else:
            self.actor = ActorNetwork(hidden_sizes[1], action_dim, hidden_sizes[2]).to(
                device
            )

        # Create critic network
        self.critic = CriticNetwork(hidden_sizes[1], hidden_sizes[2]).to(device)

        # Create optimizer with combined parameters from all networks
        params = (
            list(self.shared_network.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )

        self.optimizer = optim.Adam(params, lr=lr, eps=1e-5)

        # Setup learning rate scheduler with warm-up and cosine decay
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=1e-5
        )

        # Training metrics
        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "clip_fraction": [],
            "explained_variance": [],
            "mean_episode_rewards": [],
            "mean_episode_lengths": [],
        }

        # Ensure checkpoint directory exists if provided
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def act(self, state, deterministic=False):
        """
        Select action based on current policy.

        Args:
            state: Current state observation
            deterministic: Whether to take deterministic action (for evaluation)

        Returns:
            action: Selected action
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            features = self.shared_network(state)
            action, _, _ = self.actor.get_action(features, deterministic)

        # Convert action to numpy and apply bounds for continuous actions
        if self.continuous_actions:
            action_np = action.cpu().numpy().squeeze()
            if self.action_bounds is not None:
                low, high = self.action_bounds
                action_np = np.clip(action_np, low, high)
            return action_np
        else:
            return action.cpu().item()

    def compute_gae(self, rewards, values, next_value, dones):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE provides a balance between bias and variance in advantage estimation by
        using a weighted average of temporal difference errors.

        Args:
            rewards: Tensor of rewards [T]
            values: Tensor of state values [T]
            next_value: Value of the next state (for bootstrapping)
            dones: Tensor of done flags [T]

        Returns:
            returns: Tensor of returns (value targets) [T]
            advantages: Tensor of advantages [T]
        """
        advantages = torch.zeros_like(rewards)
        gae = 0

        # Compute advantages from the end to start
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # Calculate delta (TD error)
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            # Calculate GAE recursively
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns for value function targets (advantage + value)
        returns = advantages + values

        return returns, advantages

    def update(self, trajectory, update_metrics=True):
        """
        Update policy and value networks using PPO algorithm.

        Args:
            trajectory: Dictionary containing trajectory data
            update_metrics: Whether to update metrics history

        Returns:
            metrics: Dictionary of metrics from the update
        """
        # Start timing for measuring update time
        start_time = time.time()

        # Extract and convert trajectory data to tensors
        states = torch.FloatTensor(trajectory["states"]).to(self.device)
        actions = (
            torch.FloatTensor(trajectory["actions"]).to(self.device)
            if self.continuous_actions
            else torch.LongTensor(trajectory["actions"]).to(self.device)
        )
        rewards = torch.FloatTensor(trajectory["rewards"]).to(self.device)
        dones = torch.FloatTensor(trajectory["dones"]).to(self.device)
        next_states = torch.FloatTensor(trajectory["next_states"]).to(self.device)

        # Get episode statistics if available
        episode_rewards = trajectory.get("episode_rewards", [])
        episode_lengths = trajectory.get("episode_lengths", [])

        # Get old action log probabilities and values
        with torch.no_grad():
            old_features = self.shared_network(states)
            old_log_probs, _ = self.actor.evaluate_actions(old_features, actions)
            old_values = self.critic(old_features)

            # Get value of the last next_state for bootstrapping
            last_features = self.shared_network(next_states[-1].unsqueeze(0))
            last_value = self.critic(last_features).item()

        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, old_values, last_value, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare for mini-batch updates
        batch_size = min(self.mini_batch_size, states.shape[0])
        num_samples = states.shape[0]
        indices = np.arange(num_samples)

        # Track metrics
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "kl_divergence": 0,
            "clip_fraction": 0,
            "explained_variance": 0,
            "update_time": 0,
        }

        # Perform multiple PPO updates
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            # Process mini-batches
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Forward pass
                features = self.shared_network(batch_states)
                new_log_probs, entropy = self.actor.evaluate_actions(
                    features, batch_actions
                )
                values = self.critic(features)

                # Calculate policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                policy_loss1 = -ratio * batch_advantages
                policy_loss2 = (
                    -torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * batch_advantages
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Calculate value loss (with optional clipping of value function)
                # Uncomment the clipped version for more conservative value updates
                # values_clipped = old_values[batch_indices] + torch.clamp(
                #     values - old_values[batch_indices], -self.clip_range, self.clip_range
                # )
                # value_loss1 = F.mse_loss(values, batch_returns, reduction='none')
                # value_loss2 = F.mse_loss(values_clipped, batch_returns, reduction='none')
                # value_loss = torch.max(value_loss1, value_loss2).mean()

                # Simpler value loss without clipping
                value_loss = F.mse_loss(values, batch_returns)

                # Calculate total loss
                loss = (
                    policy_loss
                    + self.critic_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent too large updates
                torch.nn.utils.clip_grad_norm_(
                    list(self.shared_network.parameters())
                    + list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # Compute metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()

                # Compute KL divergence and clip fraction
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()
                    metrics["kl_divergence"] += kl
                    metrics["clip_fraction"] += (
                        (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                    )

            # Early stopping based on KL divergence
            avg_kl = metrics["kl_divergence"] / (
                (epoch + 1) * np.ceil(num_samples / batch_size)
            )
            if avg_kl > 1.5 * self.target_kl:
                print(
                    f"Early stopping at epoch {epoch+1}/{self.ppo_epochs} due to reaching max KL divergence ({avg_kl:.4f})."
                )
                break

        # Update learning rate
        self.scheduler.step()

        # Compute total update time
        metrics["update_time"] = time.time() - start_time

        # Average metrics
        num_updates = self.ppo_epochs * np.ceil(num_samples / batch_size)
        for k in [
            "policy_loss",
            "value_loss",
            "entropy",
            "kl_divergence",
            "clip_fraction",
        ]:
            metrics[k] /= num_updates

        # Compute explained variance
        with torch.no_grad():
            features = self.shared_network(states)
            values_pred = self.critic(features)
            var_y = torch.var(returns)
            explained_var = 1 - torch.var(returns - values_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = explained_var.item()

        # Add episode statistics if available
        if episode_rewards:
            metrics["mean_episode_reward"] = np.mean(episode_rewards)
            metrics["mean_episode_length"] = np.mean(episode_lengths)

            # Update metrics history if requested
            if update_metrics:
                self.metrics_history["mean_episode_rewards"].append(
                    metrics["mean_episode_reward"]
                )
                self.metrics_history["mean_episode_lengths"].append(
                    metrics["mean_episode_length"]
                )
                self.metrics_history["policy_loss"].append(metrics["policy_loss"])
                self.metrics_history["value_loss"].append(metrics["value_loss"])
                self.metrics_history["entropy"].append(metrics["entropy"])
                self.metrics_history["kl_divergence"].append(metrics["kl_divergence"])
                self.metrics_history["clip_fraction"].append(metrics["clip_fraction"])
                self.metrics_history["explained_variance"].append(
                    metrics["explained_variance"]
                )

        return metrics

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
                "metrics_history": self.metrics_history,
                "continuous_actions": self.continuous_actions,
                "action_bounds": self.action_bounds,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

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

        # Load metrics history if available
        if "metrics_history" in checkpoint:
            self.metrics_history = checkpoint["metrics_history"]

        # Load action space type if available
        if "continuous_actions" in checkpoint:
            assert (
                self.continuous_actions == checkpoint["continuous_actions"]
            ), "Mismatch in action space type"

        # Load action bounds if available
        if "action_bounds" in checkpoint and checkpoint["action_bounds"] is not None:
            self.action_bounds = checkpoint["action_bounds"]

        print(f"Model loaded from {filepath}")

    def plot_metrics(self, figsize=(15, 10), save_path=None):
        """
        Plot training metrics.

        Args:
            figsize: Figure size (width, height)
            save_path: Path to save the figure, if None, the figure is shown but not saved
        """
        if not self.metrics_history["policy_loss"]:
            print("No metrics to plot yet.")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Plot rewards and episode lengths
        axes[0, 0].plot(self.metrics_history["mean_episode_rewards"])
        axes[0, 0].set_title("Mean Episode Reward")
        axes[0, 0].set_xlabel("Update")

        axes[0, 1].plot(self.metrics_history["mean_episode_lengths"])
        axes[0, 1].set_title("Mean Episode Length")
        axes[0, 1].set_xlabel("Update")

        # Plot losses
        axes[0, 2].plot(self.metrics_history["policy_loss"])
        axes[0, 2].set_title("Policy Loss")
        axes[0, 2].set_xlabel("Update")

        axes[1, 0].plot(self.metrics_history["value_loss"])
        axes[1, 0].set_title("Value Loss")
        axes[1, 0].set_xlabel("Update")

        # Plot KL and entropy
        axes[1, 1].plot(self.metrics_history["kl_divergence"])
        axes[1, 1].set_title("KL Divergence")
        axes[1, 1].set_xlabel("Update")

        axes[1, 2].plot(self.metrics_history["entropy"])
        axes[1, 2].set_title("Policy Entropy")
        axes[1, 2].set_xlabel("Update")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        else:
            plt.show()

    def collect_rollout(self, env, max_steps, render=False):
        """
        Collect trajectory rollout from the environment.

        Args:
            env: Environment to interact with
            max_steps: Maximum number of steps to collect
            render: Whether to render the environment

        Returns:
            trajectory: Dictionary containing trajectory data
        """
        # Initialize trajectory data
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        log_probs = []

        # Initialize episode tracking
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        # Reset environment
        state = env.reset()

        for _ in range(max_steps):
            if render:
                env.render()

            # Select action
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)

            # Get action from policy
            with torch.no_grad():
                features = self.shared_network(state_tensor)
                action, log_prob, _ = self.actor.get_action(features)

            # Convert action to numpy for environment
            if self.continuous_actions:
                action_np = action.cpu().numpy().squeeze()
                if self.action_bounds is not None:
                    low, high = self.action_bounds
                    action_np = np.clip(action_np, low, high)
            else:
                action_np = action.cpu().item()

            # Execute action in environment
            next_state, reward, done, _ = env.step(action_np)

            # Store data
            states.append(state)
            if self.continuous_actions:
                actions.append(action.cpu().numpy().squeeze())
            else:
                actions.append(action.cpu().item())
            rewards.append(reward)
            dones.append(float(done))
            next_states.append(next_state)
            log_probs.append(log_prob.cpu().item())

            # Update episode tracking
            current_episode_reward += reward
            current_episode_length += 1

            # Handle episode termination
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                state = env.reset()
            else:
                state = next_state

        # Add remaining episode if not done
        if current_episode_length > 0:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

        # Construct trajectory dictionary
        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "next_states": np.array(next_states),
            "log_probs": np.array(log_probs),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        return trajectory
