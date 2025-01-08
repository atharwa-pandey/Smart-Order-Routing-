import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
from multiprocessing import Pool


# Generate synthetic market data
def generate_synthetic_order_book(num_ticks=1000, num_venues=3):
    data = {}
    for venue in range(num_venues):
        prices = np.random.normal(loc=100, scale=1, size=num_ticks).cumsum()
        volumes = np.random.randint(10, 100, size=num_ticks)
        timestamps = pd.date_range("2023-01-01", periods=num_ticks, freq="S")
        data[f"Venue_{venue + 1}"] = pd.DataFrame({"Timestamp": timestamps, "Price": prices, "Volume": volumes})
    return data


# RL Environment for Smart Order Routing
class SOR_Environment:
    def __init__(self, order_book_data, total_volume):
        self.order_book = order_book_data
        self.total_volume = total_volume
        self.remaining_volume = total_volume
        self.current_tick = 0
        self.num_venues = len(order_book_data)

    def reset(self):
        self.current_tick = 0
        self.remaining_volume = self.total_volume
        return self._get_state()

    def _get_state(self):
        if self.current_tick >= len(self.order_book["Venue_1"]):
            return np.zeros((self.num_venues * 2,))  # Price and volume per venue

        state = []
        for venue in self.order_book.values():
            price = venue.iloc[self.current_tick]["Price"]
            volume = venue.iloc[self.current_tick]["Volume"]
            state.extend([price, volume])
        return np.array(state)

    def step(self, actions):
        if self.current_tick >= len(self.order_book["Venue_1"]):
            return self._get_state(), -float("inf"), True, {}

        total_executed_cost = 0
        executed_volumes = []

        for venue_idx, allocated_volume in enumerate(actions):
            venue_key = f"Venue_{venue_idx + 1}"
            venue = self.order_book[venue_key]
            price = venue.iloc[self.current_tick]["Price"]
            available_volume = venue.iloc[self.current_tick]["Volume"]
            executed_volume = min(allocated_volume, available_volume, self.remaining_volume)
            executed_volumes.append(executed_volume)
            total_executed_cost += executed_volume * price
            self.remaining_volume -= executed_volume

        self.current_tick += 1
        avg_execution_price = total_executed_cost / sum(executed_volumes) if sum(executed_volumes) > 0 else 0
        benchmark_price = np.mean([venue.iloc[self.current_tick - 1]["Price"] for venue in self.order_book.values()])
        slippage = (avg_execution_price - benchmark_price) / benchmark_price * 100 if benchmark_price != 0 else 0

        reward = -avg_execution_price  # Minimize average execution price
        done = self.remaining_volume <= 0 or self.current_tick >= len(self.order_book["Venue_1"])
        return self._get_state(), reward, done, {
            "executed_volumes": executed_volumes,
            "average_price": avg_execution_price,
            "benchmark_price": benchmark_price,
            "slippage": slippage,
        }


# Double Deep Q-Learning Agent
class DDQLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay buffer
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.min_experience = 100

        # Networks
        self.eval_net = self._build_model()
        self.target_net = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.state_size,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="mse")
        return model

    def update_target_network(self):
        self.target_net.set_weights(self.eval_net.get_weights())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 100, size=self.action_size)  # Random allocation
        q_values = self.eval_net.predict(state.reshape(1, -1), verbose=0)
        return np.clip(q_values[0], 0, 100)  # Ensure non-negative allocations

    def replay(self):
        if len(self.memory) < self.min_experience:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.eval_net.predict(state.reshape(1, -1), verbose=0)
            if done:
                target[0][np.argmax(action)] = reward
            else:
                next_action = np.argmax(self.eval_net.predict(next_state.reshape(1, -1), verbose=0))
                target_q = self.target_net.predict(next_state.reshape(1, -1), verbose=0)
                target[0][np.argmax(action)] = reward + self.gamma * target_q[0][next_action]

            self.eval_net.fit(state.reshape(1, -1), target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Function for training a single environment (moved to the top level for pickling)
def train_single_env(agent, env, episodes_per_env):
    print("Single Train Started")
    state = env.reset()
    total_reward = 0
    for _ in range(episodes_per_env):
        actions = agent.act(state)
        next_state, reward, done, _ = env.step(actions)
        agent.store_experience(state, actions, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        if done:
            state = env.reset()
    return total_reward


# Parallelized Training
def parallel_train_ddql(agent, environments, episodes_per_env=100):
    print("Single Train Started")
    with Pool(len(environments)) as pool:
        results = pool.starmap(train_single_env, [(agent, env, episodes_per_env) for env in environments])
    return results


# Main Execution
if __name__ == "__main__":
    # Enable GPU optimizations
    tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")
        except RuntimeError as e:
            print(e)

    # Generate synthetic order book data
    order_book_data = generate_synthetic_order_book()

    # Train RL-based SOR with DDQL
    num_envs = 4  # Number of parallel environments
    environments = [SOR_Environment(order_book_data, total_volume=500) for _ in range(num_envs)]
    state_size = environments[0].num_venues * 2  # Price and volume for each venue
    action_size = environments[0].num_venues  # Actions per venue
    agent = DDQLAgent(state_size=state_size, action_size=action_size)

    print("Training DDQL Agent with Parallelism...")
    rewards = parallel_train_ddql(agent, environments, episodes_per_env=500 // num_envs)
    print(f"Training complete. Total rewards: {rewards}")

    # Test DDQL-based SOR
    print("\nTesting DDQL-based SOR:")
    state = environments[0].reset()
    ddql_executed_prices = []
    ddql_slippage = []
    while True:
        actions = agent.act(state)
        next_state, _, done, info = environments[0].step(actions)
        state = next_state
        ddql_executed_prices.append(info["average_price"])
        ddql_slippage.append(info["slippage"])
        if done:
            break
    ddql_avg_price = sum(ddql_executed_prices) / len(ddql_executed_prices)
    ddql_avg_slippage = sum(ddql_slippage) / len(ddql_slippage)
    print(f"DDQL Average Execution Price: {ddql_avg_price:.2f}")
    print(f"DDQL Average Slippage: {ddql_avg_slippage:.2f}%")