"""DQN agent implementation for discrete action spaces.

Contains:
- QNet: simple 3-layer MLP producing Q-values for each action.
- DQNAgent: lightweight DQN with replay buffer, target network and Adam optimizer.
"""

import copy

import numpy as np

import syna.functions as F
import syna.layers as L
from syna import Model, optim
from syna.utils.rl import ReplayBuffer


class QNet(Model):
    """Small feed-forward network that outputs Q-values for each action.

    Args:
        action_size: number of discrete actions.
    """

    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class DQNAgent:
    """Simplified DQN agent.

    The agent uses an epsilon-greedy policy, a replay buffer, and a target network.
    """

    def __init__(
        self,
        action_size: int = 2,
        gamma: float = 0.98,
        lr: float = 5e-4,
        epsilon: float = 0.1,
        buffer_size: int = 10000,
        batch_size: int = 32,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(action_size)
        self.qnet_target = copy.deepcopy(self.qnet)

        self.optimizer = optim.Adam(lr)
        self.optimizer.setup(self.qnet)

    def select_action(self, state: np.ndarray) -> int:
        """Return an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        qs = self.qnet(state[np.newaxis, :])
        return int(qs.data.argmax())

    def update(self, state, action, reward, next_state, done):
        """Store transition and update Q network from a sampled batch when available."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample()

        qs = self.qnet(state_b)
        q = qs[np.arange(self.batch_size), action_b]

        next_qs = self.qnet_target(next_state_b)
        next_q = next_qs.max(axis=1)
        next_q.unchain()

        target = reward_b + (1 - done_b) * self.gamma * next_q
        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_qnet(self):
        """Copy current Q network weights to the target network."""
        self.qnet_target = copy.deepcopy(self.qnet)
