import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, examples):
        self.buffer.extend(examples)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer  # Return all if not enough samples
        return random.sample(self.buffer, batch_size)

    def all(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)