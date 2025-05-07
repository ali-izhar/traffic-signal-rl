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
    """Advantage Actor-Critic (A2C) Agent implementation with GPU optimizations"""

    def __init__(
        self,
        input_dim,
        output_dim,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.95,
        shared_layers=2,
        shared_width=512,
        actor_layers=2,
        actor_width=256,
        critic_layers=2,
        critic_width=256,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        super().__init__(input_dim, output_dim)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        # Network architecture parameters
        self._shared_layers = shared_layers
        self._shared_width = shared_width
        self._actor_layers = actor_layers
        self._actor_width = actor_width
        self._critic_layers = critic_layers
        self._critic_width = critic_width

        # Apply GPU optimizations
        self.enable_gpu_optimizations()

        # Build actor and critic models
        self._build_models()

        # Memory for training
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []

        # Performance tracking
        self._training_steps = 0
        self._last_loss = {"actor": 0, "critic": 0, "total": 0}

    def _build_models(self):
        """Build actor and critic models with shared layers and GPU optimizations"""
        # Input layer
        inputs = keras.Input(shape=(self._input_dim,))

        # Use float16 for hidden layers on GPU for performance
        dtype = "float16" if self.gpu_available else "float32"

        # Shared layers with mixed precision for GPU
        x = layers.Dense(self._shared_width, activation="relu", dtype=dtype)(inputs)
        for _ in range(self._shared_layers - 1):
            x = layers.Dense(self._shared_width, activation="relu", dtype=dtype)(x)

        # Actor head
        actor_x = x
        for _ in range(self._actor_layers):
            actor_x = layers.Dense(self._actor_width, activation="relu", dtype=dtype)(
                actor_x
            )

        # Use float32 for output layer for numerical stability
        actor_output = layers.Dense(
            self._output_dim, activation="softmax", dtype="float32"
        )(actor_x)

        # Critic head
        critic_x = x
        for _ in range(self._critic_layers):
            critic_x = layers.Dense(self._critic_width, activation="relu", dtype=dtype)(
                critic_x
            )
        critic_output = layers.Dense(1, activation="linear", dtype="float32")(critic_x)

        # Create models
        self._actor = keras.Model(inputs=inputs, outputs=actor_output)
        self._critic = keras.Model(inputs=inputs, outputs=critic_output)

        # Use epsilon for numerical stability with mixed precision
        actor_optimizer = Adam(learning_rate=self._actor_lr, epsilon=1e-7)
        critic_optimizer = Adam(learning_rate=self._critic_lr, epsilon=1e-7)

        # Compile with JIT for better performance if on GPU
        self._actor.compile(
            optimizer=actor_optimizer, jit_compile=True if self.gpu_available else False
        )
        self._critic.compile(
            optimizer=critic_optimizer,
            loss="mse",
            jit_compile=True if self.gpu_available else False,
        )

    def act(self, state, epsilon=0):
        """Select action based on actor network probabilities with epsilon for exploration"""
        state = np.reshape(state, [1, self._input_dim]).astype(np.float32)
        action_probs = self._actor.predict(state, verbose=0)[0]

        # Apply epsilon-greedy policy
        if random.random() < epsilon:
            return random.randint(0, self._output_dim - 1)
        else:
            # Sample from action probability distribution
            # Use 'replace=False' to avoid numerical issues with float16
            action_probs = np.array(action_probs, dtype=np.float32)
            return np.random.choice(self._output_dim, p=action_probs)

    def store_experience(self, state, action, reward, next_state, done=False):
        """Add experience to memory for batch learning"""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

    def learn(self, state, action, reward, next_state, done=False):
        """Add experience to memory (for compatibility)"""
        self.store_experience(state, action, reward, next_state, done)

        # Train if episode is done (for compatibility with older code)
        if done:
            self.batch_learn()

    def batch_learn(self, epsilon=0):
        """Train actor and critic networks based on collected experiences"""
        if not self._states:
            return  # Nothing to learn

        # Convert to numpy arrays with proper types for GPU
        states = np.array(self._states, dtype=np.float32)
        actions = np.array(self._actions, dtype=np.int32)
        rewards = np.array(self._rewards, dtype=np.float32)
        next_states = np.array(self._next_states, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        # Get critic values for states and next_states efficiently
        values = self._critic.predict(states, verbose=0).flatten()
        next_values = self._critic.predict(next_states, verbose=0).flatten()

        # Calculate advantages and critic targets efficiently using numpy vectorization
        # For each step, calculate return and advantage
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # Calculate returns and advantages in reverse order
        discounted_sum = 0
        for i in reversed(range(len(rewards))):
            # If done, reset the discounted sum
            if dones[i]:
                discounted_sum = 0

            # Calculate discounted return
            discounted_sum = rewards[i] + self._gamma * discounted_sum * (1 - dones[i])
            returns[i] = discounted_sum

            # Advantage = Return - Value
            advantages[i] = returns[i] - values[i]

        # Normalize advantages for stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # Train critic - Use the calculated returns as targets
        critic_history = self._critic.fit(
            states,
            returns,
            epochs=1,
            verbose=0,
            batch_size=min(64, len(states)),
            use_multiprocessing=True,
            workers=4,
        )
        critic_loss = (
            critic_history.history["loss"][0] if "loss" in critic_history.history else 0
        )

        # Train actor using custom gradient tape
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self._actor(states, training=True)

            # Calculate action log probabilities
            action_masks = tf.one_hot(actions, self._output_dim)
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            log_probs = tf.math.log(selected_action_probs + 1e-10)

            # Policy gradient loss with advantages
            actor_loss = -tf.reduce_mean(log_probs * advantages)

            # Entropy loss for exploration
            entropy = -tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-10), axis=1
            )
            entropy_loss = -self._entropy_coef * tf.reduce_mean(entropy)

            # Combined loss
            total_loss = actor_loss + entropy_loss

        # Get and apply actor gradients
        actor_grads = tape.gradient(total_loss, self._actor.trainable_variables)
        # Apply gradient clipping to prevent exploding gradients
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
        self._actor.optimizer.apply_gradients(
            zip(actor_grads, self._actor.trainable_variables)
        )

        # Track losses
        self._last_loss = {
            "actor": float(actor_loss),
            "critic": float(critic_loss),
            "entropy": float(entropy_loss),
            "total": float(total_loss),
        }
        self._training_steps += 1

        # Clear memory after training
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []

        return self._last_loss

    def save(self, path):
        """Save actor and critic models"""
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "a2c_actor.keras"))
        self._critic.save(os.path.join(path, "a2c_critic.keras"))

        # Save configuration
        config = {
            "actor_lr": self._actor_lr,
            "critic_lr": self._critic_lr,
            "gamma": self._gamma,
            "entropy_coef": self._entropy_coef,
            "value_coef": self._value_coef,
            "shared_layers": self._shared_layers,
            "shared_width": self._shared_width,
            "actor_layers": self._actor_layers,
            "actor_width": self._actor_width,
            "critic_layers": self._critic_layers,
            "critic_width": self._critic_width,
            "training_steps": self._training_steps,
        }

        # Save config as text file
        with open(os.path.join(path, "a2c_config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

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

        # Re-enable GPU optimizations after loading
        if self.gpu_available:
            # Set JIT compilation for loaded models
            self._actor.compile(optimizer=self._actor.optimizer, jit_compile=True)
            self._critic.compile(
                optimizer=self._critic.optimizer, loss="mse", jit_compile=True
            )
