from abc import ABC, abstractmethod
import os
import numpy as np


class Agent(ABC):
    """Base class for all reinforcement learning agents"""

    def __init__(self, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim

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

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim
