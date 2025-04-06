#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Proximal Policy Optimization (PPO) Agent"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class SharedNetwork(nn.Module):
    """Shared feature extraction layers for both actor and critic"""

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
    """Policy network (Actor) for discrete action spaces"""

    def __init__(self, feature_dim, action_dim, hidden_size=64):
        super(ActorNetwork, self).__init__()

        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Initialize weights
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Compute action logits"""
        logits = self.actor_head(features)
        return logits

    def get_distribution(self, features):
        """Get action distribution"""
        logits = self.forward(features)
        return Categorical(logits=logits)

    def evaluate_actions(self, features, actions):
        """Evaluate actions and return log probabilities and entropy"""
        distribution = self.get_distribution(features)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        return log_probs, entropy

    def get_action(self, features, deterministic=False):
        """Get action, log probability, and entropy"""
        distribution = self.get_distribution(features)

        if deterministic:
            action = torch.argmax(distribution.probs, dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy().mean()

        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """Value network (Critic)"""

    def __init__(self, feature_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()

        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        # Initialize weights
        for layer in self.critic_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Compute state value"""
        return self.critic_head(features).squeeze(-1)


class PPOAgent:
    """Proximal Policy Optimization agent implementation"""

    def __init__(
        self,
        state_dim,
        action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=5e-4,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        ppo_epochs=4,
        batch_size=512,
        target_kl=0.015,
        hidden_sizes=[256, 128, 64],
        checkpoint_dir="checkpoints",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
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

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    def act(self, state, deterministic=False):
        """Select action based on current policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            features = self.shared_network(state)
            action, _, _ = self.actor.get_action(features, deterministic)

        return action.cpu().item()

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        # Compute advantages from the end to start
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns for value function targets
        returns = advantages + values

        return returns, advantages

    def update(self, trajectory):
        """Update policy and value networks using PPO algorithm"""
        # Extract and convert trajectory data to tensors
        states = torch.FloatTensor(trajectory["states"]).to(self.device)
        actions = torch.LongTensor(trajectory["actions"]).to(self.device)
        rewards = torch.FloatTensor(trajectory["rewards"]).to(self.device)
        dones = torch.FloatTensor(trajectory["dones"]).to(self.device)
        next_states = torch.FloatTensor(trajectory["next_states"]).to(self.device)

        # Get old action log probabilities
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
        batch_size = min(self.batch_size, states.shape[0])
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

                # Calculate value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Calculate total loss
                loss = (
                    policy_loss
                    + self.critic_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
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
                metrics["entropy"] += entropy.item()

                # Compute KL divergence
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
                    f"Early stopping at epoch {epoch+1}/{self.ppo_epochs} due to reaching max KL divergence."
                )
                break

        # Update learning rate
        self.scheduler.step()

        # Average metrics
        num_updates = self.ppo_epochs * np.ceil(num_samples / batch_size)
        for k in metrics.keys():
            metrics[k] /= num_updates

        # Compute explained variance
        with torch.no_grad():
            features = self.shared_network(states)
            values_pred = self.critic(features)
            var_y = torch.var(returns)
            explained_var = 1 - torch.var(returns - values_pred) / var_y
            metrics["explained_variance"] = explained_var.item()

        return metrics

    def save(self, filename):
        """Save agent state"""
        torch.save(
            {
                "shared_network": self.shared_network.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            os.path.join(self.checkpoint_dir, filename),
        )

    def load(self, filename):
        """Load agent state"""
        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, filename), map_location=self.device
        )
        self.shared_network.load_state_dict(checkpoint["shared_network"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
