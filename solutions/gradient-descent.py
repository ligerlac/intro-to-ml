import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Generate noisy data on a straight line
n_points = 50
true_w = 2.5  # True slope
true_b = 1.0  # True intercept
noise_std = 1.8

# Generate x values and center them around zero to decorrelate w and b
x = np.linspace(0, 10, n_points)
x = x - np.mean(x)  # Center x around zero for more circular loss contours
# Generate y values with noise
y = true_w * x + true_b + np.random.normal(0, noise_std, n_points)

# Create ranges for w and b to plot loss surface
w_range = np.linspace(true_w - 1, true_w + 1, 50)
b_range = np.linspace(true_b - 2, true_b + 2, 50)
W, B = np.meshgrid(w_range, b_range)

# Calculate loss (Mean Squared Error) for each combination of w and b
def calculate_loss(w, b, x_data, y_data):
    y_pred = w * x_data + b
    return np.mean((y_data - y_pred) ** 2)

# Vectorized loss calculation
Loss = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Loss[i, j] = calculate_loss(W[i, j], B[i, j], x, y)

# Perform gradient descent to get optimization path
def gradient_descent_path(x_data, y_data, learning_rate=0.05, n_iterations=20):
    """Perform gradient descent and return the path"""
    # Initialize parameters away from optimum for better visualization
    w_gd = true_w - 0.8
    b_gd = true_b + 1.
    
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
        w_gd -= learning_rate * dw
        b_gd -= learning_rate * db
        
        # Store path
        w_path.append(w_gd)
        b_path.append(b_gd)
        loss_path.append(calculate_loss(w_gd, b_gd, x_data, y_data))
    
    return np.array(w_path), np.array(b_path), np.array(loss_path)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss (MSE)')

# change angle for better view
ax.view_init(elev=17, azim=-70)

plt.tight_layout()

plt.savefig(f'gradient_descent_path.png', dpi=300)


for i in range(10):
    # Get gradient descent path
    w_path, b_path, loss_path = gradient_descent_path(x, y, learning_rate=0.03, n_iterations=i)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')

    # Plot gradient descent path as connected points
    # ax2.plot(w_path, b_path, loss_path, 'ro-', markersize=4, linewidth=2, alpha=0.8, label='Gradient descent path')

    # Add arrows showing gradient descent steps
    for i in range(len(w_path)-1):
        ax.quiver(w_path[i], b_path[i], loss_path[i],
                w_path[i+1] - w_path[i], 
                b_path[i+1] - b_path[i], 
                loss_path[i+1] - loss_path[i],
                color='red', arrow_length_ratio=0.1, alpha=1)

    # Mark important points
    ax.scatter([w_path[0]], [b_path[0]], [loss_path[0]], 
            color='orange', s=100, marker='s', label='Start point')

    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Bias (b)')
    ax.set_zlabel('Loss (MSE)')

    # change angle for better view
    ax.view_init(elev=17, azim=-70)

    plt.tight_layout()

    plt.savefig(f'gradient_descent_path_{i}.png', dpi=300)


exit(0)



# # Create the plots
# fig = plt.figure(figsize=(15, 6))

# # Plot 1: Original data and fitted line
# ax1 = fig.add_subplot(121)
# ax1.scatter(x, y, alpha=0.6, label='Noisy data', color='blue')
# ax1.plot(x, true_w * x + true_b, 'g--', linewidth=2, label=f'True line (w={true_w}, b={true_b})')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('Linear Regression on Noisy Data')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Plot 2: 3D Loss Surface with gradient descent arrows
# ax2 = fig.add_subplot(122, projection='3d')
# surface = ax2.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')

# # Plot gradient descent path as connected points
# # ax2.plot(w_path, b_path, loss_path, 'ro-', markersize=4, linewidth=2, alpha=0.8, label='Gradient descent path')

# # Add arrows showing gradient descent steps
# for i in range(len(w_path)-1):
#     ax2.quiver(w_path[i], b_path[i], loss_path[i],
#                w_path[i+1] - w_path[i], 
#                b_path[i+1] - b_path[i], 
#                loss_path[i+1] - loss_path[i],
#                color='red', arrow_length_ratio=0.1, alpha=1)

# # Mark important points
# ax2.scatter([w_path[0]], [b_path[0]], [loss_path[0]], 
#            color='orange', s=100, marker='s', label='Start point')
# # ax2.scatter([true_w], [true_b], [calculate_loss(true_w, true_b, x, y)], 
# #            color='green', s=100, marker='o', label='True parameters')

# ax2.set_xlabel('Weight (w)')
# ax2.set_ylabel('Bias (b)')
# ax2.set_zlabel('Loss (MSE)')
# ax2.set_title('3D Loss Surface with Gradient Descent')
# ax2.legend()

# # change angle for better view
# ax2.view_init(elev=17, azim=-70)

# plt.tight_layout()
# plt.show()
