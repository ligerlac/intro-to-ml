import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate noisy data on a straight line
n_points = 50
price_per_sqm = 12_345
minimum_price = 100_000

noise_scale = 0.1  # Scale of noise relative to the prices

sizes = np.random.uniform(80, 500, n_points)
prices = price_per_sqm * sizes + minimum_price

# add random noise and random scaling
prices = prices + np.random.normal(0, noise_scale * prices, n_points)

plt.scatter(sizes, prices, label='Data points', color='blue')
plt.show()

# save to txt with numpy
np.savetxt('data/housing_prices.txt', np.column_stack((sizes, prices)), delimiter=',', header='size,price', comments='')
