#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # kill warning about tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from .base import Agent


class DQNAgent(Agent):
    """Deep Q-Network Agent implementation"""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=4,
        width=400,
        batch_size=100,
        learning_rate=0.001,
        gamma=0.75,
        memory_size_max=50000,
        memory_size_min=600,
    ):
        super().__init__(input_dim, output_dim)
        self._num_layers = num_layers
        self._width = width
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._model = self._build_model()

        # Experience replay memory
        self._memory = []
        self._memory_size_max = memory_size_max
        self._memory_size_min = memory_size_min

    def _build_model(self):
        """Build and compile a fully connected deep neural network"""
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(self._width, activation="relu")(inputs)
        for _ in range(self._num_layers):
            x = layers.Dense(self._width, activation="relu")(x)
        outputs = layers.Dense(self._output_dim, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="dqn_model")
        model.compile(
            loss=losses.MeanSquaredError(),
            optimizer=Adam(learning_rate=self._learning_rate),
        )
        return model

    def act(self, state, epsilon=0):
        """Select an action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self._output_dim - 1)  # Explore
        else:
            return np.argmax(self.predict_one(state))  # Exploit

    def predict_one(self, state):
        """Predict Q-values for a single state"""
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    def predict_batch(self, states):
        """Predict Q-values for a batch of states"""
        return self._model.predict(states, verbose=0)

    def add_memory(self, state, action, reward, next_state, done=False):
        """Add experience to memory"""
        self._memory.append((state, action, reward, next_state, done))
        if len(self._memory) > self._memory_size_max:
            self._memory.pop(0)  # Remove oldest memory

    def learn(self, state, action, reward, next_state, done=False):
        """Add experience to memory and train if enough samples"""
        self.add_memory(state, action, reward, next_state, done)

        if len(self._memory) < self._memory_size_min:
            return  # Not enough samples

        # Sample a batch from memory
        self.replay(self._batch_size)

    def replay(self, batch_size):
        """Train model on a batch of experiences from memory"""
        if len(self._memory) < self._memory_size_min:
            return

        # Sample batch
        if batch_size > len(self._memory):
            batch = random.sample(self._memory, len(self._memory))
        else:
            batch = random.sample(self._memory, batch_size)

        states = np.array([experience[0] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])

        # Predict current Q-values and next Q-values
        q_values = self.predict_batch(states)
        next_q_values = self.predict_batch(next_states)

        # Setup training data
        x = np.zeros((len(batch), self._input_dim))
        y = np.zeros((len(batch), self._output_dim))

        # Update Q-values using Bellman equation
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = q_values[i].copy()
            if done:
                target[action] = reward
            else:
                target[action] = reward + self._gamma * np.max(next_q_values[i])

            x[i] = state
            y[i] = target

        # Train the model
        self._model.fit(x, y, epochs=1, verbose=0)

    def save(self, path):
        """Save the model to the specified path"""
        os.makedirs(path, exist_ok=True)
        self._model.save(os.path.join(path, "dqn_model.keras"))
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

        if os.path.isfile(model_path_keras):
            self._model = load_model(model_path_keras)
        elif os.path.isfile(model_path_h5):
            self._model = load_model(model_path_h5)
        else:
            raise FileNotFoundError(f"No model file found at {path}")

    @property
    def batch_size(self):
        return self._batch_size
