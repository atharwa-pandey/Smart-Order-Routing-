import numpy as np

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
            return tuple([(0, 0)] * self.num_venues)

        state = []
        for venue in self.order_book.values():
            price = venue.iloc[self.current_tick]["Price"]
            volume = venue.iloc[self.current_tick]["Volume"]
            state.append((price, volume))
        return tuple(state)

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