#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # kill warning about tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from .base import Agent


class PPOAgent(Agent):
    """Proximal Policy Optimization (PPO) Agent implementation with GPU optimizations for RTX 4090"""

    def __init__(
        self,
        input_dim,
        output_dim,
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.95,
        lambd=0.95,
        clip_ratio=0.2,
        shared_layers=3,
        shared_width=512,
        actor_layers=2,
        actor_width=256,
        critic_layers=2,
        critic_width=256,
        epochs=10,
        batch_size=512,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        super().__init__(input_dim, output_dim)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma
        self._lambd = lambd
        self._clip_ratio = clip_ratio
        self._epochs = epochs
        self._batch_size = batch_size
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

        # Memory for training - use a more efficient data structure
        self._buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],  # Store log probabilities for importance sampling
            "values": [],  # Store values for GAE calculation
        }

        # Track performance
        self._training_steps = 0
        self._last_loss = {"actor": 0, "critic": 0, "total": 0}

    def _build_models(self):
        """Build actor and critic models with shared layers and GPU optimizations"""
        # Input layer
        inputs = keras.Input(shape=(self._input_dim,))

        # Use float16 for hidden layers when on GPU for better performance
        dtype = "float16" if self.gpu_available else "float32"

        # Shared layers with mixed precision for GPU
        x = layers.Dense(self._shared_width, activation="relu", dtype=dtype)(inputs)
        for _ in range(self._shared_layers - 1):
            x = layers.Dense(self._shared_width, activation="relu", dtype=dtype)(x)

        # Actor head - policy network
        actor_x = x
        for _ in range(self._actor_layers):
            actor_x = layers.Dense(self._actor_width, activation="relu", dtype=dtype)(
                actor_x
            )

        # Always use float32 for output layer for numerical stability
        actor_output = layers.Dense(
            self._output_dim, activation="softmax", dtype="float32"
        )(actor_x)

        # Critic head - value network
        critic_x = x
        for _ in range(self._critic_layers):
            critic_x = layers.Dense(self._critic_width, activation="relu", dtype=dtype)(
                critic_x
            )
        critic_output = layers.Dense(1, activation="linear", dtype="float32")(critic_x)

        # Create models
        self._actor = keras.Model(inputs=inputs, outputs=actor_output)
        self._critic = keras.Model(inputs=inputs, outputs=critic_output)

        # Optimizers with epsilon for numerical stability with mixed precision
        actor_optimizer = Adam(learning_rate=self._actor_lr, epsilon=1e-7)
        critic_optimizer = Adam(learning_rate=self._critic_lr, epsilon=1e-7)

        # Compile models with jit_compile for better performance
        self._actor.compile(
            optimizer=actor_optimizer, jit_compile=True if self.gpu_available else False
        )
        self._critic.compile(
            optimizer=critic_optimizer,
            loss="mse",
            jit_compile=True if self.gpu_available else False,
        )

    def act(self, state, epsilon=0):
        """Select action based on actor network probabilities and save log probability"""
        state = np.reshape(state, [1, self._input_dim]).astype(np.float32)
        action_probs = self._actor.predict(state, verbose=0)[0]

        # Get critic value for GAE calculation
        value = float(self._critic.predict(state, verbose=0)[0, 0])

        # Apply epsilon-greedy for additional exploration if needed
        if random.random() < epsilon:
            action = random.randint(0, self._output_dim - 1)
        else:
            # Sample from action probability distribution
            action_probs = np.array(
                action_probs, dtype=np.float32
            )  # Ensure float32 precision
            action = np.random.choice(self._output_dim, p=action_probs)

        # Store log probability for PPO update
        log_prob = np.log(action_probs[action] + 1e-10)

        # Store value for advantage calculation
        self._buffer["log_probs"].append(log_prob)
        self._buffer["values"].append(value)

        return action

    def store_experience(self, state, action, reward, next_state, done=False):
        """Add experience to buffer for batch learning"""
        self._buffer["states"].append(state)
        self._buffer["actions"].append(action)
        self._buffer["rewards"].append(reward)
        self._buffer["next_states"].append(next_state)
        self._buffer["dones"].append(done)

    def learn(self, state, action, reward, next_state, done=False):
        """Store experience and train if episode is done (for compatibility)"""
        self.store_experience(state, action, reward, next_state, done)

        # Train if episode is done
        if done:
            self.batch_learn()

    def _compute_advantages(self, rewards, values, dones, next_values):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            # For last step, use next_value if not done
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else next_values[t]
            else:
                next_value = values[t + 1]

            # Delta: R + gamma * V(s') - V(s)
            delta = rewards[t] + self._gamma * next_value * (1 - dones[t]) - values[t]

            # GAE computation: sum of discounted TD errors
            gae = delta + self._gamma * self._lambd * (1 - dones[t]) * gae
            advantages[t] = gae

            # Return = advantage + value
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def batch_learn(self, epsilon=0):
        """Train actor and critic networks with batched operations"""
        # Check if we have enough experience to train
        if not self._buffer["states"] or len(self._buffer["states"]) != len(
            self._buffer["log_probs"]
        ):
            return  # Nothing to learn or log_probs mismatch

        # Track time for performance monitoring
        start_time = time.time()

        # Convert to numpy arrays with proper types for GPU
        states = np.array(self._buffer["states"], dtype=np.float32)
        actions = np.array(self._buffer["actions"], dtype=np.int32)
        rewards = np.array(self._buffer["rewards"], dtype=np.float32)
        next_states = np.array(self._buffer["next_states"], dtype=np.float32)
        dones = np.array(self._buffer["dones"], dtype=np.float32)
        old_log_probs = np.array(self._buffer["log_probs"], dtype=np.float32)
        old_values = np.array(self._buffer["values"], dtype=np.float32)

        # Calculate next values for advantage calculation
        next_values = self._critic.predict(next_states, verbose=0).flatten()

        # Compute advantages and returns
        advantages, returns = self._compute_advantages(
            rewards, old_values, dones, next_values
        )

        # Normalize advantages for training stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Create dataset for mini-batch training - more efficient on GPU
        # Use tf.data.Dataset for better GPU utilization
        buffer_size = len(states)
        dataset = tf.data.Dataset.from_tensor_slices(
            (states, actions, old_log_probs, advantages, returns)
        )

        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size).batch(self._batch_size)

        # Calculate number of batches
        n_batches = np.ceil(buffer_size / self._batch_size).astype(np.int32)

        # Train for multiple epochs (PPO typically uses multiple epochs per batch)
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for epoch in range(self._epochs):
            for batch in dataset:
                (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                ) = batch

                # Train critic
                with tf.GradientTape() as tape:
                    # Get value predictions
                    values = self._critic(batch_states, training=True)
                    values = tf.squeeze(values, axis=1)

                    # Compute value loss
                    critic_loss = (
                        tf.reduce_mean(tf.square(values - batch_returns))
                        * self._value_coef
                    )

                # Apply critic gradients
                critic_grads = tape.gradient(
                    critic_loss, self._critic.trainable_variables
                )
                # Clip gradients for stability
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)
                self._critic.optimizer.apply_gradients(
                    zip(critic_grads, self._critic.trainable_variables)
                )

                # Train actor using PPO clipping objective
                with tf.GradientTape() as tape:
                    # Get new action probabilities
                    new_action_probs = self._actor(batch_states, training=True)

                    # Create action masks and extract new log probs
                    action_masks = tf.one_hot(batch_actions, self._output_dim)
                    responsible_outputs = tf.reduce_sum(
                        new_action_probs * action_masks, axis=1
                    )
                    new_log_probs = tf.math.log(responsible_outputs + 1e-10)

                    # Calculate ratios for importance sampling
                    ratios = tf.exp(new_log_probs - batch_old_log_probs)

                    # PPO clipped objective
                    clipped_ratios = tf.clip_by_value(
                        ratios, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio
                    )
                    policy_reward = tf.minimum(
                        ratios * batch_advantages, clipped_ratios * batch_advantages
                    )
                    policy_loss = -tf.reduce_mean(policy_reward)

                    # Add entropy term to encourage exploration
                    entropy = -tf.reduce_sum(
                        new_action_probs * tf.math.log(new_action_probs + 1e-10), axis=1
                    )
                    entropy_loss = -self._entropy_coef * tf.reduce_mean(entropy)

                    # Combined loss
                    actor_loss = policy_loss + entropy_loss

                # Apply actor gradients
                actor_grads = tape.gradient(actor_loss, self._actor.trainable_variables)
                # Clip gradients for stability
                actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
                self._actor.optimizer.apply_gradients(
                    zip(actor_grads, self._actor.trainable_variables)
                )

                # Record losses
                actor_losses.append(float(policy_loss))
                critic_losses.append(float(critic_loss))
                entropy_losses.append(float(entropy_loss))

        # Record training stats
        self._training_steps += 1

        # Calculate average losses
        self._last_loss = {
            "actor": np.mean(actor_losses),
            "critic": np.mean(critic_losses),
            "entropy": np.mean(entropy_losses),
            "total": np.mean(actor_losses)
            + np.mean(critic_losses)
            + np.mean(entropy_losses),
            "training_time": time.time() - start_time,
        }

        # Clear memory
        for key in self._buffer:
            self._buffer[key] = []

        return self._last_loss

    def save(self, path):
        """Save actor and critic models"""
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "ppo_actor.keras"))
        self._critic.save(os.path.join(path, "ppo_critic.keras"))

        # Save configuration
        config = {
            "actor_lr": self._actor_lr,
            "critic_lr": self._critic_lr,
            "gamma": self._gamma,
            "lambd": self._lambd,
            "clip_ratio": self._clip_ratio,
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
        with open(os.path.join(path, "ppo_config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

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

        # Re-enable GPU optimizations after loading
        if self.gpu_available:
            # Set JIT compilation for loaded models
            self._actor.compile(optimizer=self._actor.optimizer, jit_compile=True)
            self._critic.compile(
                optimizer=self._critic.optimizer, loss="mse", jit_compile=True
            )
