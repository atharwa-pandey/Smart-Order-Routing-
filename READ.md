**Smart Order Routing (SOR) Framework**

This project implements a Smart Order Routing (SOR) framework for evaluating order execution strategies in financial markets. The framework includes a Reinforcement Learning (RL)-based agent and compares its performance against the Time-Weighted Average Price (TWAP) benchmark strategy.


**Project Structure**

1. **data_generation.py** <br />
	•	Generates synthetic market data for multiple trading venues. <br />
Features:<br />
	•	Simulates price and volume data using stochastic models.<br />
	•	Outputs realistic Limit Order Book (LOB) data for testing strategies.

2. **environment.py** <br />
	•	Defines the SOR environment for Reinforcement Learning.<br />
Features:<br />
	•	Models multi-venue trading dynamics.<br />
	•	Tracks state, actions, and rewards for the RL agent.<br />
	•	Provides execution metrics such as average price and slippage.<br />

3. **q_agent.py** <br />
	•	Implements a Q-Learning agent for optimizing SOR decisions.<br />
Features:<br />
	•	Uses a Q-table for state-action mapping.<br />
	•	Learns optimal routing strategies through exploration and exploitation.<br />
	•	Updates Q-values based on rewards from the environment.<br />

4. **twap.py** <br />
	•	Implements the TWAP benchmark strategy.<br />
Features:<br />
	•	Splits orders evenly across fixed time intervals.<br />
	•	Provides baseline metrics for comparison with the RL agent.<br />

5. **main.py** <br />
	•	Entry point for training and testing the SOR framework.<br />
Features:
	•	Trains the Q-Learning agent over multiple episodes.<br />
	•	Tests the RL agent and TWAP strategy on unseen data.<br />
	•	Outputs performance metrics (e.g., execution price, slippage).<br />

6. **TRY_double_deep_q_agent.py** <br />
	•	Prototype implementation of the Double Deep Q-Learning (DDQL) agent.<br />
Features:<br />
	•	Uses neural networks for value approximation.<br />
	•	Improves stability with target and evaluation networks.<br />
	•	Suitable for advanced execution strategies.<br />

7. **TRY_parallel_computation.py** <br />
	•	Prototype implementation for parallelizing RL training.<br />
Features:<br />
	•	Explores multiprocessing to accelerate agent training.<br />
	•	Leverages multiple environments for simultaneous experience collection.<br />