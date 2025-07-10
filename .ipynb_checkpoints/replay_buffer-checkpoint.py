from collections import deque
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, trajectory, priority):
        max_priority = self.priorities.max() if self.buffer else 1.0
        new_priority = priority ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
            self.priorities[len(self.buffer) - 1] = max(max_priority, new_priority)
        else:
            # 找到优先级最低的位置
            min_priority_index = np.argmin(self.priorities)
            if new_priority > self.priorities[min_priority_index]:
                # 覆盖优先级最低的样本
                self.buffer[min_priority_index] = trajectory
                self.priorities[min_priority_index] = new_priority

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # total = len(self.buffer)
        # weights = (total * probabilities[indices]) ** (-beta)
        # weights /= weights.max()  # Normalize for stability

        return samples, indices
