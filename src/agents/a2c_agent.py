#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Advantage Actor-Critic (A2C) Agent"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


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
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.LayerNorm(action_dim),
        )

        # Initialize weights with smaller values
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Compute action logits"""
        return self.actor_head(features)

    def get_action_and_log_prob(self, features, deterministic=False):
        """Get action and log probability"""
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
    """Value network (Critic)"""

    def __init__(self, feature_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()

        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize weights with smaller values
        for layer in self.critic_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Compute state value"""
        return self.critic_head(features)


class A2CAgent:
    """Advantage Actor-Critic agent with entropy regularization"""

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
        checkpoint_dir="checkpoints",
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
            action, _, _ = self.actor.get_action_and_log_prob(features, deterministic)

        return action.cpu().item()

    def compute_gae(self, rewards, values, next_value, dones, gamma, gae_lambda):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]

            delta = rewards[i] + gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def update(self, states, actions, rewards, next_states, dones):
        """Update policy and value networks"""
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Extract features
        features = self.shared_network(states)
        next_features = self.shared_network(next_states)

        # Get values
        values = self.critic(features).squeeze(-1)
        next_values = self.critic(next_features).squeeze(-1)

        # Compute advantages using GAE
        advantages = self.compute_gae(
            rewards.cpu().numpy(),
            values.detach().cpu().numpy(),
            next_values[-1].detach().cpu().numpy(),
            dones.cpu().numpy(),
            self.gamma,
            self.gae_lambda,
        )

        # Convert advantages to tensor and normalize
        advantages = torch.FloatTensor(advantages).to(self.device)
        if len(advantages) > 1:  # Only normalize if we have more than one sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get log probabilities and entropy
        _, log_probs, entropy = self.actor.get_action_and_log_prob(features, False)

        # Compute returns for value loss
        returns = advantages + values.detach()

        # Calculate losses with gradient clipping
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)

        # Total loss with entropy regularization
        total_loss = (
            actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.shared_network.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

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
