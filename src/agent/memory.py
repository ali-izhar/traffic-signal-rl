"""Experience Replay Memory for Reinforcement Learning Agents"""

from typing import List, Tuple
import random


class Memory:
    """Experience replay buffer that stores and samples transitions for training.

    - Let D be the replay buffer with capacity N
    - D = {e_1, e_2, ..., e_N} where e_i is an experience tuple (s_t, a_t, r_t, s_{t+1})
    - At each training step, a mini-batch of size B is sampled uniformly: {e_j ~ U(D)}_{j=1}^B

    Attributes:
        _samples: List of experience tuples (state, action, reward, next_state, done)
        _size_max: Maximum capacity of the replay buffer
        _size_min: Minimum number of samples required before sampling begins
    """

    def __init__(self, size_max: int, size_min: int) -> None:
        """Initialize the replay memory.

        Args:
            size_max: Maximum capacity of the replay buffer
            size_min: Minimum number of samples required before sampling begins
        """
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample: Tuple) -> None:
        """Add a sample (experience tuple) to the memory.
        When buffer exceeds maximum capacity, oldest samples are removed (FIFO).

        Args:
            sample: Experience tuple (state, action, reward, next_state, done)
        """
        self._samples.append(sample)
        if len(self._samples) > self._size_max:
            self._samples.pop(0)  # Remove oldest sample when capacity is exceeded

    def get_samples(self, n: int) -> List[Tuple]:
        """Sample n experiences randomly from the replay buffer.
        Uniform random sampling follows the principle:
        P(e_i) = 1/|D| for all experiences e_i in the buffer D

        Args:
            n: Number of samples to retrieve

        Returns:
            List of randomly sampled experience tuples, or empty list if
            buffer size is below minimum threshold
        """
        if len(self._samples) < self._size_min:
            return []

        return random.sample(self._samples, min(n, len(self._samples)))

    @property
    def size(self) -> int:
        """Current number of samples in the memory.

        Returns:
            Integer representing current buffer size
        """
        return len(self._samples)

    @property
    def ready(self) -> bool:
        """Check if memory has enough samples to begin training.

        Returns:
            Boolean indicating if buffer size exceeds minimum threshold
        """
        return len(self._samples) >= self._size_min
