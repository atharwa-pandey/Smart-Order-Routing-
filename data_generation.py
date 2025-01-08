import numpy as np
import pandas as pd

# Generate synthetic market data
def generate_synthetic_order_book(num_ticks=1000, num_venues=3):
    data = {}
    for venue in range(num_venues):
        prices = np.random.normal(loc=100, scale=1, size=num_ticks).cumsum()
        volumes = np.random.randint(10, 100, size=num_ticks)
        timestamps = pd.date_range("2023-01-01", periods=num_ticks, freq="S")
        data[f"Venue_{venue + 1}"] = pd.DataFrame({"Timestamp": timestamps, "Price": prices, "Volume": volumes})
    return data