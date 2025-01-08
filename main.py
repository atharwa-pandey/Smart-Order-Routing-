from data_generation import generate_synthetic_order_book
from environment import SOR_Environment
from q_agent import QLearningAgent, train_rl_agent
from twap import simulate_twap_strategy

# Main Execution
if __name__ == "__main__":
    # Generate synthetic order book data
    order_book_data = generate_synthetic_order_book()

    # Train RL-based SOR
    environment = SOR_Environment(order_book_data, total_volume=500)
    agent = QLearningAgent(num_venues=environment.num_venues)
    print("Training RL Agent...")
    train_rl_agent(environment, agent, episodes=500)

    # Test RL-based SOR
    print("\nTesting RL-based SOR:")
    state = environment.reset()
    rl_executed_prices = []
    rl_slippage = []
    while True:
        actions = agent.choose_action(state)
        next_state, _, done, info = environment.step(actions)
        state = next_state
        rl_executed_prices.append(info["average_price"])
        rl_slippage.append(info["slippage"])
        if done:
            break
    rl_avg_price = sum(rl_executed_prices) / len(rl_executed_prices)
    rl_avg_slippage = sum(rl_slippage) / len(rl_slippage)
    print(f"RL Average Execution Price: {rl_avg_price:.2f}")
    print(f"RL Average Slippage: {rl_avg_slippage:.2f}%")

    # TWAP Benchmark
    print("\nSimulating TWAP Strategy...")
    twap_metrics = simulate_twap_strategy(order_book_data, total_volume=500, time_window=10)
    print(f"TWAP Average Execution Price: {twap_metrics['Average Execution Price']:.2f}")
    print(f"TWAP Average Slippage: {twap_metrics['Average Slippage (%)']:.2f}%")