import numpy as np
from collections import deque
import gymnasium as gym
import Agents.Models.tensor_cnn as cnn
import os, csv, random

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
        self.model = cnn.build_model(state_shape, n_actions, lr)
        self.memory = ReplayBuffer(buffer_size)
        self.model_path = "Experiments/dqn_model.keras"
        self.log_path = "Experiments/dqn_log.csv"

    def get_action(self, state):
        """
        Epsilon-greedy action selection:
        - With probability ε: choose random action
        - Otherwise: choose action with highest Q-value
        """
        if random.random() < self.epsilon:
            return random.choices([1, 2, 3, 4, 5], weights=[0.00, 0.2, 0.2, 0.6, 0.0])[0] 
        q_vals = self.model.predict(state[np.newaxis], verbose=0)
        return int(np.argmax(q_vals[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.add(s, a, r, s2, done)
    
    def save_model(self):
        cnn.save_model(self.model, self.model_path)

    def load_model(self):
        self.model = cnn.load_model(self.model_path, lr=1e-4)
        
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

    def train(self, env, episodes=1, render=False, max_steps=100):
        #loading the log
        rewards = []
        start_ep = 0
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    rewards.append(float(row[1]))
                start_ep = len(rewards)
                print("Resuming from episode", start_ep)
                #loading the model          
                if os.path.exists(self.model_path):
                    self.load_model()
                    print("Model loaded successfully.")
        
        for ep in range(episodes):
            current_ep = start_ep + ep
            state, _ = env.reset()
            done = False
            step = 0
            ep_reward = 0
            while not done and step < max_steps:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                ep_reward += reward
                step += 1
                if render and step % 10 == 0:
                    env.render()  
                print(f"Step {step}, Action: {action}, Reward: {reward:.1f}, Epsilon: {self.epsilon:.2f}")
            rewards.append(ep_reward)
            print(f"Episode {current_ep}, Total Reward: {ep_reward:.1f}, Epsilon: {self.epsilon:.2f}")
            
            #implementing epsilon decay, to ensure model uses its training to learn
            if self.epsilon > 0.05 and current_ep % 5 == 0:
                self.epsilon *= 0.98
            
            if current_ep % 10 == 0:
                self.save_model()
                print(f"Model saved at episode {ep+1}")
                with open(self.log_path, 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Episode", "Reward"])
                    for i, r in enumerate(rewards):
                        writer.writerow([i, r])
                    print(f"Log saved at episode {ep+1}")