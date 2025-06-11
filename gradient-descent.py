import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import load_house_prices, plot_housing_prices, plot_fit_landscape_and_loss

# load the housing prices data
sizes, prices = load_house_prices('data/housing_prices.txt')

# plot the data
plot_housing_prices(sizes, prices)

# scale the data such that the features are centered around zero and have reasonable range
####### YOUR CODE HERE #######
x = ...
y = ...
####### END OF YOUR CODE ######

# plot the scaled data
plot_housing_prices(x, y, scaled=True)

# Create ranges for w and b to plot loss surface
w_range = np.linspace(0, 2, 50)
b_range = np.linspace(-0.5, 0.5, 50)
W, B = np.meshgrid(w_range, b_range)

# Calculate loss (Mean Squared Error) for each combination of w and b
def calculate_loss(w, b, x_data, y_data):
    ######### YOUR CODE HERE #########
    y_pred = ...
    loss = ...
    ######## END OF YOUR CODE ########
    return loss

# Calcuate loss for each combination of w and b
Loss = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Loss[i, j] = calculate_loss(W[i, j], B[i, j], x, y)

# Perform gradient descent to get optimization path
def gradient_descent_path(x_data, y_data, learning_rate=0.05, n_iterations=20):
    """Perform gradient descent and return the path"""
    # Initialize parameters away from optimum for better visualization
    w_gd = 0.4
    b_gd = 0.4
    
    # Store path
    w_path = [w_gd]
    b_path = [b_gd]
    loss_path = [calculate_loss(w_gd, b_gd, x_data, y_data)]
    
    for i in range(n_iterations):
        # Calculate predictions
        y_pred = w_gd * x_data + b_gd
        
        # Calculate gradients
        dw = -2 * np.mean(x_data * (y_data - y_pred))
        db = -2 * np.mean(y_data - y_pred)
        
        # Update parameters
        ######### YOUR CODE HERE #########
        w_gd -= ...
        b_gd -= ...
        ######## END OF YOUR CODE ########
        
        # Store path
        w_path.append(w_gd)
        b_path.append(b_gd)
        loss_path.append(calculate_loss(w_gd, b_gd, x_data, y_data))
    
    return np.array(w_path), np.array(b_path), np.array(loss_path)


# Plot the fit landscape and loss for different iterations of gradient descent
for i in range(50):
    w_path, b_path, loss_path = gradient_descent_path(x, y, learning_rate=0.1, n_iterations=i)
    plot_fit_landscape_and_loss(W, B, Loss, x, y, w_path, b_path, loss_path)
