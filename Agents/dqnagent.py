import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym


class ReplayBuffer:
    """
    Stores experiences (s, a, r, s2, done) so the agent can learn from past events.
    """
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_shape, n_actions,
                 gamma=0.99, epsilon=1.0, lr=1e-4,
                 batch_size=32, buffer_size=50000):
        self.n_actions = n_actions
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # exploration probability
        self.batch_size = batch_size
        self.model = self.build_model(state_shape, n_actions, lr)
        self.memory = ReplayBuffer(buffer_size)

    def build_model(self, input_shape, n_actions, lr):
        """
        Builds a simple CNN for input images.
        Input: (3, 48, 48) → Output: vector of size 8 (Q-values)
        """
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 8, strides=4, activation='relu'),
            layers.Conv2D(64, 4, strides=2, activation='relu'),
            layers.Conv2D(64, 3, strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_actions)  # output Q-values for each action
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='mean_squared_error')
        return model

    def get_action(self, state):
        """
        Epsilon-greedy action selection:
        - With probability ε: choose random action
        - Otherwise: choose action with highest Q-value
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_vals = self.model.predict(state[np.newaxis], verbose=0)
        return int(np.argmax(q_vals[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.add(s, a, r, s2, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to learn

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_targets = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next[i])
            q_targets[i][actions[i]] = target

        self.model.fit(states, q_targets, epochs=1, verbose=0)

    def train(self, env, episodes=1000, render=False):
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                ep_reward += reward
                if render:
                    env.render()  
            print(f"Episode {ep+1}, Total Reward: {ep_reward:.1f}")