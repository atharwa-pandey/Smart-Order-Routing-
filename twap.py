import numpy as np

# TWAP Strategy Simulation
def simulate_twap_strategy(order_book_data, total_volume, time_window):
    interval_volume = total_volume // time_window
    executed_prices, executed_volumes = [], []
    remaining_volume = total_volume
    slippages = []

    for tick in range(time_window):
        best_venue, best_price = None, float('inf')
        for venue_name, venue_data in order_book_data.items():
            price = venue_data.iloc[tick]["Price"]
            if price < best_price:
                best_price = price
                best_venue = venue_name

        venue_data = order_book_data[best_venue]
        available_volume = venue_data.iloc[tick]["Volume"]
        executed_volume = min(interval_volume, available_volume, remaining_volume)
        remaining_volume -= executed_volume
        executed_prices.append(best_price)
        executed_volumes.append(executed_volume)

        # Calculate slippage for this step
        benchmark_price = np.mean([venue_data.iloc[tick]["Price"] for venue_data in order_book_data.values()])
        slippage = (best_price - benchmark_price) / benchmark_price * 100 if benchmark_price != 0 else 0
        slippages.append(slippage)

        if remaining_volume <= 0:
            break

    avg_price = sum(p * v for p, v in zip(executed_prices, executed_volumes)) / sum(executed_volumes)
    avg_slippage = sum(slippages) / len(slippages)
    return {"Average Execution Price": avg_price, "Average Slippage (%)": avg_slippage}
