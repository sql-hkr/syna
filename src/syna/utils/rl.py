"""Reinforcement learning helpers.

Simple utilities for RL experiments: a ReplayBuffer and a Trainer class
used in DQN-style example scripts.

Trainer expectations:
    - The provided `agent` must implement `select_action(state)`, `update(...)`,
        and `sync_qnet()` for periodic target network syncs. Trainer keeps a
        simple reward history and plots at the end of training.
"""

import random
from collections import deque
from typing import Deque, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class ReplayBuffer:
    """
    Simple FIFO replay buffer for storing transitions.

    Args:
        buffer_size: maximum number of transitions to store.
        batch_size: number of transitions returned by sample().
    """

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=buffer_size
        )
        self.batch_size = batch_size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self):
        """
        Sample a minibatch of transitions.
        If there are fewer than batch_size transitions available, sample what's present.
        Returns:
            state, action, reward, next_state, done (numpy arrays)
        """
        n = min(self.batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch]).astype(np.int32)
        return states, actions, rewards, next_states, dones


class Trainer:
    """
    Trainer for running episodes with a provided agent and environment.

    Args:
        env_name: gymnasium environment id.
        num_episodes: number of training episodes.
        sync_interval: how often (episodes) to call agent.sync_qnet().
        epsilon: epsilon value set on agent for the final evaluation run.
        agent: object implementing select_action(state), update(...), and sync_qnet().
    """

    def __init__(
        self,
        env_name="CartPole-v1",
        num_episodes=300,
        sync_interval=20,
        epsilon=0.1,
        agent=None,
    ):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.sync_interval = sync_interval
        self.epsilon = epsilon
        self.agent = agent
        # create environment; render_mode kept for compatibility with .render() usage later
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.reward_history = []

    def train(self):
        """Run training loop, plot rewards, and run one final evaluation episode."""
        for episode in range(1, self.num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                # gymnasium step returns (obs, reward, terminated, truncated, info)
                done_flag = done or truncated

                self.agent.update(state, action, reward, next_state, done_flag)
                state = next_state
                total_reward += reward

            if episode % self.sync_interval == 0:
                self.agent.sync_qnet()

            self.reward_history.append(total_reward)
            if episode % 10 == 0:
                print(f"episode: {episode}, total reward: {total_reward}")

        # plot training curve
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.plot(self.reward_history)
        plt.show()

        # final evaluation with deterministic epsilon
        if self.agent is not None:
            self.agent.epsilon = self.epsilon
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                state = next_state
                total_reward += reward
                # note: render_mode="rgb_array" may not display a window in all setups
                try:
                    self.env.render()
                except Exception:
                    pass
            print("Total Reward (evaluation):", total_reward)
        return self.reward_history
