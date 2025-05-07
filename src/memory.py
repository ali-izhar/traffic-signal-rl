import random
import numpy as np
from collections import deque


class Memory:
    def __init__(self, size_max, size_min):
        self._samples = deque(
            maxlen=size_max
        )  # Using deque with maxlen for automatic size management
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        Add a sample into the memory
        """
        self._samples.append(sample)
        # No need to manually remove elements as deque with maxlen handles this automatically

    def get_samples(self, n):
        """
        Get n samples randomly from the memory with optimized numpy operations
        """
        current_size = self._size_now()
        if current_size < self._size_min:
            return []

        if n > current_size:
            samples = list(self._samples)  # get all the samples
        else:
            # Use numpy for faster random sampling
            indices = np.random.choice(current_size, size=n, replace=False)
            samples = [list(self._samples)[i] for i in indices]

        return samples

    def _size_now(self):
        """
        Check how full the memory is
        """
        return len(self._samples)


class PrioritizedMemory(Memory):
    """Extended memory with prioritized experience replay for more efficient learning"""

    def __init__(self, size_max, size_min, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(size_max, size_min)
        self._priorities = deque(maxlen=size_max)
        self._alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self._beta = beta  # Importance sampling correction
        self._beta_increment = beta_increment  # Beta annealing
        self._epsilon = 1e-5  # Small value to avoid zero priority

    def add_sample(self, sample, error=None):
        """Add sample with priority based on TD error"""
        max_priority = max(self._priorities) if self._priorities else 1.0
        if error is not None:
            priority = (abs(error) + self._epsilon) ** self._alpha
        else:
            priority = max_priority

        self._samples.append(sample)
        self._priorities.append(priority)

    def get_samples(self, n):
        """Get samples based on their priorities"""
        current_size = self._size_now()
        if current_size < self._size_min:
            return [], [], []

        # Increase beta over time for more accurate bias correction
        self._beta = min(1.0, self._beta + self._beta_increment)

        # Convert priorities to probabilities
        priorities = np.array(list(self._priorities))
        probabilities = priorities / np.sum(priorities)

        # Sample based on priorities
        indices = np.random.choice(
            current_size, min(n, current_size), p=probabilities, replace=False
        )

        # Calculate importance sampling weights
        weights = (current_size * probabilities[indices]) ** (-self._beta)
        weights /= np.max(weights)  # Normalize weights

        samples = [list(self._samples)[i] for i in indices]
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        """Update priorities based on new TD errors"""
        for idx, error in zip(indices, errors):
            if idx < len(self._priorities):
                self._priorities[idx] = (abs(error) + self._epsilon) ** self._alpha
