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
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from .base import Agent
from memory import Memory, PrioritizedMemory


class DQNAgent(Agent):
    """Deep Q-Network Agent implementation with optimizations for high-end GPUs"""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=4,
        width=512,
        batch_size=512,
        learning_rate=0.001,
        gamma=0.75,
        memory_size_max=100000,
        memory_size_min=1000,
        use_prioritized_replay=False,
        use_double_dqn=False,
        target_update_freq=1000,
    ):
        super().__init__(input_dim, output_dim)
        self._num_layers = num_layers
        self._width = width
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._use_double_dqn = use_double_dqn
        self._target_update_freq = target_update_freq
        self._steps = 0

        # Apply GPU optimizations
        self.enable_gpu_optimizations()

        # Build models
        self._model = self._build_model()

        # For double DQN, create a target network
        if self._use_double_dqn:
            self._target_model = self._build_model()
            self._update_target_model()

        # Experience replay memory
        self._use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self._memory = PrioritizedMemory(memory_size_max, memory_size_min)
        else:
            self._memory = Memory(memory_size_max, memory_size_min)

        self._memory_size_max = memory_size_max
        self._memory_size_min = memory_size_min

    def _build_model(self):
        """Build and compile a fully connected deep neural network with GPU optimizations"""
        inputs = keras.Input(shape=(self._input_dim,))

        # Use float16 for hidden layers on GPU for performance
        dtype = "float16" if self.gpu_available else "float32"

        x = layers.Dense(self._width, activation="relu", dtype=dtype)(inputs)
        for _ in range(self._num_layers):
            x = layers.Dense(self._width, activation="relu", dtype=dtype)(x)

        # Output layer always uses float32 for numerical stability
        outputs = layers.Dense(self._output_dim, activation="linear", dtype="float32")(
            x
        )

        model = keras.Model(inputs=inputs, outputs=outputs, name="dqn_model")

        # Use epsilon param to avoid numerical instability with mixed precision
        optimizer = Adam(learning_rate=self._learning_rate, epsilon=1e-7)

        model.compile(
            loss=losses.MeanSquaredError(),
            optimizer=optimizer,
            # Enable JIT compilation for improved performance
            jit_compile=True if self.gpu_available else False,
        )
        return model

    def _update_target_model(self):
        """Update target model weights for double DQN"""
        if self._use_double_dqn:
            self._target_model.set_weights(self._model.get_weights())
            print("Target model updated")

    def act(self, state, epsilon=0):
        """Select an action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self._output_dim - 1)  # Explore
        else:
            return np.argmax(self.predict_one(state))  # Exploit

    def predict_one(self, state):
        """Predict Q-values for a single state"""
        state = np.reshape(state, [1, self._input_dim]).astype(np.float32)
        return self._model.predict(state, verbose=0)

    def predict_batch(self, states):
        """Predict Q-values for a batch of states"""
        return self._model.predict(states.astype(np.float32), verbose=0)

    def store_experience(self, state, action, reward, next_state, done=False):
        """Store experience in memory for batch learning"""
        if self._use_prioritized_replay:
            # For prioritized replay, calculate initial TD error for priority
            current_q = self.predict_one(state)[0][action]
            next_q = np.max(self.predict_one(next_state)[0])
            target_q = reward + (1 - done) * self._gamma * next_q
            td_error = abs(current_q - target_q)
            self._memory.add_sample((state, action, reward, next_state, done), td_error)
        else:
            self._memory.add_sample((state, action, reward, next_state, done))

        # Update step counter
        self._steps += 1

        # Update target network periodically if using double DQN
        if self._use_double_dqn and self._steps % self._target_update_freq == 0:
            self._update_target_model()

    def learn(self, state, action, reward, next_state, done=False):
        """Add experience to memory (forward to store_experience for compatibility)"""
        self.store_experience(state, action, reward, next_state, done)

        # If not using batch_learn, train on each sample
        if len(self._memory._samples) >= self._memory_size_min:
            self._train_on_batch(self._batch_size)

    def batch_learn(self, epsilon=0):
        """Process all stored experiences in efficient batches"""
        if len(self._memory._samples) < self._memory_size_min:
            return

        # Train on multiple batches for more efficient GPU utilization
        self._train_on_batch(self._batch_size)

    def _train_on_batch(self, batch_size):
        """Train the model on a batch of experiences"""
        if len(self._memory._samples) < self._memory_size_min:
            return

        # Sample batch with prioritization if enabled
        if self._use_prioritized_replay:
            batch, indices, weights = self._memory.get_samples(batch_size)
            importance_weights = np.array(weights, dtype=np.float32)
        else:
            batch = self._memory.get_samples(batch_size)
            indices = None
            importance_weights = None

        if not batch:
            return

        # Extract batch components
        states = np.array([experience[0] for experience in batch], dtype=np.float32)
        actions = np.array([experience[1] for experience in batch], dtype=np.int32)
        rewards = np.array([experience[2] for experience in batch], dtype=np.float32)
        next_states = np.array(
            [experience[3] for experience in batch], dtype=np.float32
        )
        dones = np.array([experience[4] for experience in batch], dtype=np.float32)

        # Compute current Q values and target Q values efficiently
        current_q_values = self.predict_batch(states)

        if self._use_double_dqn:
            # Double DQN: use online network to select actions, target network to evaluate
            next_q_values = self._target_model.predict(next_states, verbose=0)
            next_actions = np.argmax(
                self._model.predict(next_states, verbose=0), axis=1
            )
            next_q_values_selected = next_q_values[
                np.arange(len(next_actions)), next_actions
            ]
        else:
            # Standard DQN
            next_q_values = self.predict_batch(next_states)
            next_q_values_selected = np.max(next_q_values, axis=1)

        # Compute target Q values
        target_q_values = current_q_values.copy()
        for i in range(len(batch)):
            target_q_values[i, actions[i]] = rewards[i]
            if not dones[i]:
                target_q_values[i, actions[i]] += (
                    self._gamma * next_q_values_selected[i]
                )

        # Train with importance sampling weights if using prioritized replay
        if self._use_prioritized_replay and importance_weights is not None:
            # Compute TD errors for updating priorities
            td_errors = np.abs(
                target_q_values[np.arange(len(actions)), actions]
                - current_q_values[np.arange(len(actions)), actions]
            )

            # Create sample weights for the fit function
            sample_weights = importance_weights

            # Update priorities in the memory
            self._memory.update_priorities(indices, td_errors)

            # Train with sample weights
            self._model.fit(
                states,
                target_q_values,
                sample_weight=sample_weights,
                batch_size=self._batch_size,
                epochs=1,
                verbose=0,
                use_multiprocessing=True,
                workers=4,
            )
        else:
            # Standard training
            self._model.fit(
                states,
                target_q_values,
                batch_size=self._batch_size,
                epochs=1,
                verbose=0,
                use_multiprocessing=True,
                workers=4,
            )

    def save(self, path):
        """Save the model to the specified path"""
        os.makedirs(path, exist_ok=True)
        self._model.save(os.path.join(path, "dqn_model.keras"))

        # Save target model if using double DQN
        if self._use_double_dqn:
            self._target_model.save(os.path.join(path, "dqn_target_model.keras"))

        # Save configuration
        config = {
            "use_double_dqn": self._use_double_dqn,
            "use_prioritized_replay": self._use_prioritized_replay,
            "target_update_freq": self._target_update_freq,
            "gamma": self._gamma,
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "num_layers": self._num_layers,
            "width": self._width,
        }

        # Save config as text file for reference
        with open(os.path.join(path, "dqn_config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        try:
            plot_model(
                self._model,
                to_file=os.path.join(path, "model_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception as e:
            print(f"Warning: Could not generate model visualization: {e}")

    def load(self, path):
        """Load the model from the specified path"""
        model_path_keras = os.path.join(path, "dqn_model.keras")
        model_path_h5 = os.path.join(path, "dqn_model.h5")

        target_path_keras = os.path.join(path, "dqn_target_model.keras")
        target_path_h5 = os.path.join(path, "dqn_target_model.h5")

        # Load main model
        if os.path.isfile(model_path_keras):
            self._model = load_model(model_path_keras)
        elif os.path.isfile(model_path_h5):
            self._model = load_model(model_path_h5)
        else:
            raise FileNotFoundError(f"No model file found at {path}")

        # Load target model if using double DQN
        if self._use_double_dqn:
            if os.path.isfile(target_path_keras):
                self._target_model = load_model(target_path_keras)
            elif os.path.isfile(target_path_h5):
                self._target_model = load_model(target_path_h5)
            elif self._model:  # If no target model found but main model exists, copy it
                self._target_model = self._build_model()
                self._target_model.set_weights(self._model.get_weights())
                print("Created target model from main model")

    @property
    def batch_size(self):
        return self._batch_size
