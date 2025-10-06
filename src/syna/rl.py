import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def sample(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class Trainer:
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
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.agent = agent
        self.reward_history = []

    def train(self):
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            if episode % self.sync_interval == 0:
                self.agent.sync_qnet()

            self.reward_history.append(total_reward)
            if episode % 10 == 0:
                print("episode :{}, total reward : {}".format(episode, total_reward))

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.show()

        self.agent.epsilon = self.epsilon
        state, _ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            self.env.render()
        print("Total Reward:", total_reward)
