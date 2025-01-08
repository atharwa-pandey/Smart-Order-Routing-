import numpy as np
import random
from collections import defaultdict

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, num_venues, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.num_venues = num_venues
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(num_venues))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, 100, size=self.num_venues)  # Random allocation of volumes
        q_values = self.q_table[str(state)]
        return (q_values / np.sum(q_values)) * 100  # Normalize Q-values to allocate percentages

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[str(state)][action]
        max_next_q = np.max(self.q_table[str(next_state)])
        target_q = reward + (self.gamma * max_next_q * (1 - done))
        self.q_table[str(state)][action] += self.lr * (target_q - current_q)


# Training the RL Agent
def train_rl_agent(environment, agent, episodes=500):
    for episode in range(episodes):
        state = environment.reset()
        total_reward = 0

        while True:
            actions = agent.choose_action(state)
            next_state, reward, done, _ = environment.step(actions)
            agent.learn(state, np.argmax(actions), reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
