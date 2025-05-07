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


class A2CAgent(Agent):
    """Advantage Actor-Critic (A2C) Agent implementation"""

    def __init__(
        self,
        input_dim,
        output_dim,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.95,
        shared_layers=2,
        shared_width=256,
        actor_layers=1,
        actor_width=128,
        critic_layers=1,
        critic_width=128,
    ):
        super().__init__(input_dim, output_dim)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma

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
        """Select action based on actor network probabilities"""
        state = np.reshape(state, [1, self._input_dim])
        action_probs = self._actor.predict(state, verbose=0)[0]

        # Apply epsilon-greedy on top of policy probabilities
        if random.random() < epsilon:
            return random.randint(0, self._output_dim - 1)
        else:
            # Sample from action probability distribution
            return np.random.choice(self._output_dim, p=action_probs)

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
        """Train actor and critic networks based on collected experiences"""
        if not self._states:
            return  # Nothing to learn

        # Convert to numpy arrays
        states = np.array(self._states)
        actions = np.array(self._actions)
        rewards = np.array(self._rewards)
        next_states = np.array(self._next_states)
        dones = np.array(self._dones)

        # Get critic values for states and next_states
        values = self._critic.predict(states, verbose=0).flatten()
        next_values = self._critic.predict(next_states, verbose=0).flatten()

        # Calculate advantages and critic targets
        returns = []
        advantages = []
        discounted_reward = 0

        for i in reversed(range(len(rewards))):
            # If done, reset the discounted reward
            if dones[i]:
                discounted_reward = 0

            # Calculate discounted return and advantage
            discounted_reward = rewards[i] + self._gamma * discounted_reward * (
                1 - dones[i]
            )
            returns.append(discounted_reward)

            # Advantage = Return - Value
            advantage = discounted_reward - values[i]
            advantages.append(advantage)

        # Reverse the lists to maintain time order
        returns = np.array(returns[::-1])
        advantages = np.array(advantages[::-1])

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # Train critic
        self._critic.fit(states, returns, epochs=1, verbose=0)

        # Train actor using custom gradient tape
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self._actor(states, training=True)

            # One-hot encode actions
            actions_one_hot = tf.one_hot(actions, self._output_dim)

            # Selected action probabilities
            selected_action_probs = tf.reduce_sum(
                action_probs * actions_one_hot, axis=1
            )

            # Calculate loss: -log(π(a|s)) * advantage
            actor_loss = -tf.math.log(selected_action_probs + 1e-10) * advantages
            actor_loss = tf.reduce_mean(actor_loss)

            # Add entropy bonus to encourage exploration
            entropy = -tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-10), axis=1
            )
            entropy_bonus = 0.01 * tf.reduce_mean(entropy)

            # Final actor loss
            loss = actor_loss - entropy_bonus

        # Apply gradients to actor model
        grads = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.trainable_variables)
        )

        # Clear memory
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []

    def save(self, path):
        """Save actor and critic models"""
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "a2c_actor.keras"))
        self._critic.save(os.path.join(path, "a2c_critic.keras"))

        try:
            plot_model(
                self._actor,
                to_file=os.path.join(path, "actor_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
            plot_model(
                self._critic,
                to_file=os.path.join(path, "critic_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception as e:
            print(f"Warning: Could not generate model visualization: {e}")

    def load(self, path):
        """Load actor and critic models"""
        actor_path = os.path.join(path, "a2c_actor.keras")
        critic_path = os.path.join(path, "a2c_critic.keras")

        actor_path_h5 = os.path.join(path, "a2c_actor.h5")
        critic_path_h5 = os.path.join(path, "a2c_critic.h5")

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
