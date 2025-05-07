#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # kill warning about tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from .base import Agent


class PPOAgent(Agent):
    """Proximal Policy Optimization (PPO) Agent implementation"""

    def __init__(
        self,
        input_dim,
        output_dim,
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.95,
        lambd=0.95,
        clip_ratio=0.2,
        shared_layers=2,
        shared_width=256,
        actor_layers=1,
        actor_width=128,
        critic_layers=1,
        critic_width=128,
        epochs=10,
        batch_size=64,
    ):
        super().__init__(input_dim, output_dim)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma
        self._lambd = lambd
        self._clip_ratio = clip_ratio
        self._epochs = epochs
        self._batch_size = batch_size

        # Network architecture parameters
        self._shared_layers = shared_layers
        self._shared_width = shared_width
        self._actor_layers = actor_layers
        self._actor_width = actor_width
        self._critic_layers = critic_layers
        self._critic_width = critic_width

        # Build actor and critic models
        self._build_models()

        # Memory for training
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []
        self._log_probs = []  # For importance sampling ratio calculation

    def _build_models(self):
        """Build actor and critic models with shared layers"""
        # Input layer
        inputs = keras.Input(shape=(self._input_dim,))

        # Shared layers
        x = layers.Dense(self._shared_width, activation="relu")(inputs)
        for _ in range(self._shared_layers - 1):
            x = layers.Dense(self._shared_width, activation="relu")(x)

        # Actor head
        actor_x = x
        for _ in range(self._actor_layers):
            actor_x = layers.Dense(self._actor_width, activation="relu")(actor_x)
        actor_output = layers.Dense(self._output_dim, activation="softmax")(actor_x)

        # Critic head
        critic_x = x
        for _ in range(self._critic_layers):
            critic_x = layers.Dense(self._critic_width, activation="relu")(critic_x)
        critic_output = layers.Dense(1, activation="linear")(critic_x)

        # Create models
        self._actor = keras.Model(inputs=inputs, outputs=actor_output)
        self._critic = keras.Model(inputs=inputs, outputs=critic_output)

        # Compile models
        self._actor.compile(optimizer=Adam(learning_rate=self._actor_lr))
        self._critic.compile(optimizer=Adam(learning_rate=self._critic_lr), loss="mse")

    def act(self, state, epsilon=0):
        """Select action based on actor network probabilities and save log probability"""
        state = np.reshape(state, [1, self._input_dim])
        action_probs = self._actor.predict(state, verbose=0)[0]

        # Apply epsilon-greedy on top of policy probabilities if needed
        if random.random() < epsilon:
            action = random.randint(0, self._output_dim - 1)
        else:
            # Sample from action probability distribution
            action = np.random.choice(self._output_dim, p=action_probs)

        # Save log probability of the selected action for PPO update
        log_prob = np.log(action_probs[action] + 1e-10)
        self._log_probs.append(log_prob)

        return action

    def add_memory(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

    def learn(self, state, action, reward, next_state, done=False):
        """Add experience to memory and train if episode is done"""
        self.add_memory(state, action, reward, next_state, done)

        if done:
            self.train()

    def train(self):
        """Train actor and critic networks based on collected experiences using PPO"""
        if not self._states or len(self._states) != len(self._log_probs):
            return  # Nothing to learn or mismatch in log probs

        # Convert to numpy arrays
        states = np.array(self._states)
        actions = np.array(self._actions)
        rewards = np.array(self._rewards)
        next_states = np.array(self._next_states)
        dones = np.array(self._dones)
        old_log_probs = np.array(self._log_probs)

        # Get critic values for states
        values = self._critic.predict(states, verbose=0).flatten()

        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = (
                    0
                    if dones[t]
                    else self._critic.predict(np.array([next_states[t]]), verbose=0)[
                        0, 0
                    ]
                )
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self._gamma * next_value * (1 - dones[t]) - values[t]

            # GAE calculation
            gae = delta + self._gamma * self._lambd * (1 - dones[t]) * gae
            advantages[t] = gae

            # Returns for critic training
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # Update policy using multiple epochs of gradient descent
        for _ in range(self._epochs):
            # Optionally use mini-batches
            if self._batch_size < len(states):
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), self._batch_size):
                    end = start + self._batch_size
                    if end > len(states):
                        break
                    mb_indices = indices[start:end]
                    self._update_batch(
                        states[mb_indices],
                        actions[mb_indices],
                        old_log_probs[mb_indices],
                        advantages[mb_indices],
                        returns[mb_indices],
                    )
            else:
                # Use all data at once if batch size is larger than dataset
                self._update_batch(states, actions, old_log_probs, advantages, returns)

        # Clear memory
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []
        self._log_probs = []

    def _update_batch(self, states, actions, old_log_probs, advantages, returns):
        """Update policy and value function for a batch of data"""
        # Train critic
        self._critic.fit(states, returns, epochs=1, verbose=0)

        # Train actor using custom gradient tape for PPO clipping
        with tf.GradientTape() as tape:
            # Get new action probabilities
            action_probs = self._actor(states, training=True)

            # Convert actions to one-hot
            actions_one_hot = tf.one_hot(actions, self._output_dim)

            # New log probabilities
            new_log_probs = tf.reduce_sum(
                tf.math.log(action_probs + 1e-10) * actions_one_hot, axis=1
            )

            # Ratio between old and new policy
            ratio = tf.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            clip_1 = ratio * advantages
            clip_2 = (
                tf.clip_by_value(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
                * advantages
            )

            # Take minimum to clip the objective
            actor_loss = -tf.reduce_mean(tf.minimum(clip_1, clip_2))

            # Add entropy bonus for exploration
            entropy = -tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-10), axis=1
            )
            entropy_bonus = 0.01 * tf.reduce_mean(entropy)

            # Final loss
            loss = actor_loss - entropy_bonus

        # Apply gradients to actor model
        grads = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.trainable_variables)
        )

    def save(self, path):
        """Save actor and critic models"""
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "ppo_actor.keras"))
        self._critic.save(os.path.join(path, "ppo_critic.keras"))

        try:
            plot_model(
                self._actor,
                to_file=os.path.join(path, "ppo_actor_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
            plot_model(
                self._critic,
                to_file=os.path.join(path, "ppo_critic_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception as e:
            print(f"Warning: Could not generate model visualization: {e}")

    def load(self, path):
        """Load actor and critic models"""
        actor_path = os.path.join(path, "ppo_actor.keras")
        critic_path = os.path.join(path, "ppo_critic.keras")

        actor_path_h5 = os.path.join(path, "ppo_actor.h5")
        critic_path_h5 = os.path.join(path, "ppo_critic.h5")

        # Try loading .keras format first, then .h5
        if os.path.isfile(actor_path):
            self._actor = load_model(actor_path)
        elif os.path.isfile(actor_path_h5):
            self._actor = load_model(actor_path_h5)
        else:
            raise FileNotFoundError(f"No actor model found at {path}")

        if os.path.isfile(critic_path):
            self._critic = load_model(critic_path)
        elif os.path.isfile(critic_path_h5):
            self._critic = load_model(critic_path_h5)
        else:
            raise FileNotFoundError(f"No critic model found at {path}")
