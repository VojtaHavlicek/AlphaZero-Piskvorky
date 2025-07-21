#!/usr/bin/env python3
"""
Filename: replay_buffer.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Buffer to store training examples from self-play games.
License: MIT
"""


import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, game_examples):
        """
        Add a list of game examples to the buffer.
        FIFO behavior is maintained by using a deque with a maximum length.

        Examples should be in the format:
        

        Args:
            game_examples (list): List of game examples to add.
        """
        self.buffer.extend(game_examples)

    def sample_batch(self, batch_size):
        """
        Sample a batch of training examples from the buffer.

        TODO: should return states/policies/values separately? 
        """
        if len(self.buffer) < batch_size:
            return self.buffer  # Return all if not enough samples
        return random.sample(self.buffer, batch_size)

    def all(self) -> list:
        """
        Get all examples in the buffer.

        Returns:
            list: List of all examples in the buffer.
        """
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)
    