from abc import ABC, abstractmethod
import os
import numpy as np
import tensorflow as tf


class Agent(ABC):
    """Base class for all reinforcement learning agents"""

    def __init__(self, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._gpu_available = len(tf.config.list_physical_devices("GPU")) > 0

    @abstractmethod
    def act(self, state, epsilon=0):
        """
        Select an action based on the current state

        Args:
            state: Current state representation
            epsilon: Exploration parameter (if applicable)

        Returns:
            The selected action
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done=False):
        """
        Update the agent's knowledge based on experience

        Args:
            state: Current state representation
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: Whether the episode is done
        """
        pass

    def store_experience(self, state, action, reward, next_state, done=False):
        """
        Store experience for batch learning
        Default implementation forwards to learn
        Agents can override this to implement more efficient batch learning

        Args:
            state: Current state representation
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: Whether the episode is done
        """
        self.learn(state, action, reward, next_state, done)

    def batch_learn(self, epsilon=0):
        """
        Process stored experiences in batches (more efficient on GPU)
        Default implementation does nothing
        Agents that store experience should override this

        Args:
            epsilon: Exploration parameter (if applicable)
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save the agent to the specified path

        Args:
            path: Directory to save the agent
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load the agent from the specified path

        Args:
            path: Directory to load the agent from
        """
        pass

    def enable_gpu_optimizations(self):
        """
        Configure the agent to use GPU optimizations if available
        """
        if self._gpu_available:
            # Set memory growth to avoid memory fragmentation
            for device in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass

            # Use mixed precision if possible - improves performance on newer GPUs
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

            return True
        return False

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def gpu_available(self):
        return self._gpu_available
