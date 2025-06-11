import numpy as np
import matplotlib.pyplot as plt


def load_house_prices(file_path):
    """Load house prices from a CSV file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    sizes = data[:, 0]
    prices = data[:, 1]
    return sizes, prices


def plot_housing_prices(sizes, prices, scaled=False):
    """Plot house prices."""
    plt.figure(figsize=(6, 4))
    plt.scatter(sizes, prices, label='Data points', color='blue')
    plt.title('House Prices vs Size')
    if scaled:
        plt.xlabel('Size [a.u.]')
        plt.ylabel('Price [a.u.]')
    else:
        plt.xlabel('Size [sqm]')
        plt.ylabel('Price [CHF]')
    plt.legend()
    plt.grid()
    plt.show()


def plot_fit_landscape_and_loss(W, B, Loss, x, y, w_path, b_path, loss_path):
    # Create the plots with three subplots
    fig = plt.figure(figsize=(20, 6))   

    # Plot 1: Original data and fitted line
    ax1 = fig.add_subplot(131)
    ax1.scatter(x, y, alpha=0.6, label='Scaled data', color='blue')
    # Plot the final fitted line
    final_w, final_b = w_path[-1], b_path[-1]
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = final_w * x_line + final_b
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted line (w={final_w:.3f}, b={final_b:.3f})')
    ax1.set_xlabel('Scaled House Size')
    ax1.set_ylabel('Scaled House Price')
    ax1.set_title('Linear Regression on Scaled Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: 3D Loss Surface with gradient descent path
    ax2 = fig.add_subplot(132, projection='3d')
    surface = ax2.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')

    # Plot gradient descent path as connected points
    ax2.plot(w_path, b_path, loss_path, 'ro-', markersize=4, linewidth=2, alpha=0.8, label='Gradient descent path')

    # Add arrows showing gradient descent steps
    for i in range(len(w_path)-1):
        ax2.quiver(w_path[i], b_path[i], loss_path[i],
                w_path[i+1] - w_path[i],
                b_path[i+1] - b_path[i],
                loss_path[i+1] - loss_path[i],
                color='red', arrow_length_ratio=0.1, alpha=0.7)

    # Mark important points
    ax2.scatter([w_path[0]], [b_path[0]], [loss_path[0]],
            color='orange', s=100, marker='s', label='Start point')
    ax2.scatter([w_path[-1]], [b_path[-1]], [loss_path[-1]],
            color='green', s=100, marker='o', label='End point')

    ax2.set_xlabel('Weight (w)')
    ax2.set_ylabel('Bias (b)')
    ax2.set_zlabel('Loss (MSE)')
    ax2.set_title('3D Loss Surface with Gradient Descent')
    ax2.legend()
    # change angle for better view
    ax2.view_init(elev=17, azim=-70)

    # Plot 3: Loss value as a function of steps
    ax3 = fig.add_subplot(133)
    steps = np.arange(len(loss_path))
    ax3.plot(steps, loss_path, 'b-o', linewidth=2, markersize=4, label='Loss')
    ax3.set_xlabel('Iteration Step')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Loss Convergence During Training')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Add text showing final loss
    ax3.text(0.7, 0.9, f'Final Loss: {loss_path[-1]:.4f}', 
            transform=ax3.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()
