import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.mixture import GaussianMixture

# Set a clean, aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def plot_train_metrics(metrics: Dict[str, List[float]], plot_cost: bool = False):
    """
    Plot training metrics using Matplotlib.
    
    Args:
        metrics (dict): Dictionary containing 'train_loss' and optionally 'train_cost'.
        plot_cost (bool): Whether to plot the transport cost.
    """
    if not metrics.get('train_loss'):
        raise ValueError("Training metrics are empty.")

    epochs = range(1, len(metrics['train_loss']) + 1)
    
    cols = 2 if plot_cost else 1
    fig, ax = plt.subplots(1, cols, figsize=(12 if plot_cost else 8, 5))
    
    if not plot_cost:
        ax = [ax] # Make it iterable

    # Plot Loss
    ax[0].plot(epochs, metrics['train_loss'], label='Train Loss', color='#1f77b4')
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot Cost
    if plot_cost and 'train_cost' in metrics:
        ax[1].plot(epochs, metrics['train_cost'], label='Train Cost', color='#ff7f0e')
        ax[1].set_title("Transport Cost")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Cost")
        ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_image_transport(
    input_image: torch.Tensor, 
    transported_image: torch.Tensor, 
    target_image: torch.Tensor,
    title_suffix: str = ""
):
    """
    Visualize image transport results.
    
    Args:
        input_image (torch.Tensor): Original image tensor [C, H, W].
        transported_image (torch.Tensor): Transported image tensor [C, H, W].
        target_image (torch.Tensor): Target image tensor [C, H, W].
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Helper to convert tensor to numpy image
    def to_img(t):
        return t.permute(1, 2, 0).detach().cpu().numpy()

    ax[0].imshow(to_img(input_image))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(to_img(transported_image))
    ax[1].set_title(f'Transported Colors {title_suffix}')
    ax[1].axis('off')

    ax[2].imshow(to_img(target_image))
    ax[2].set_title('Target Colors')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_gradient_field(
    gradient_func: torch.nn.Module, 
    model: torch.nn.Module, 
    model_name: str = 'Model', 
    grid_size: int = 20
):
    """
    Plot true and predicted gradient fields.
    """
    x1 = torch.linspace(0, 1, grid_size)
    x2 = torch.linspace(0, 1, grid_size)
    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')
    x_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)

    # Compute gradients
    true_grad_grid = gradient_func(x_grid)
    with torch.no_grad():
        pred_grad_grid = model(x_grid)

    # Reshape
    true_grad_grid = true_grad_grid.reshape(grid_size, grid_size, 2)
    pred_grad_grid = pred_grad_grid.reshape(grid_size, grid_size, 2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].quiver(x1_grid, x2_grid, true_grad_grid[:, :, 0], true_grad_grid[:, :, 1], color='#1f77b4')
    ax[0].set_title("True Gradient Field")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")

    ax[1].quiver(x1_grid, x2_grid, pred_grad_grid[:, :, 0], pred_grad_grid[:, :, 1], color='#d62728')
    ax[1].set_title(f"Predicted Gradient Field ({model_name})")
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")

    plt.tight_layout()
    plt.show()

def plot_error_maps(
    gradients: List[torch.nn.Module], 
    target_func: torch.nn.Module, 
    labels: List[str], 
    xi_res: int = 100, 
    yi_res: int = 100
):
    """
    Plots error maps for multiple gradients side by side.
    """
    xi = torch.linspace(0, 1, xi_res)
    yi = torch.linspace(0, 1, yi_res)
    Xi, Yi = torch.meshgrid(xi, yi, indexing="ij")
    space = torch.cat([torch.reshape(Xi, (-1, 1)), torch.reshape(Yi, (-1, 1))], dim=1)

    targ = target_func(space)

    n_plots = len(gradients)
    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        ax = [ax]

    max_error = 0
    error_maps = []

    for i, grad in enumerate(gradients):
        # Handle ICNN special case where we need autograd
        # Ideally this logic should be outside plotting, but keeping it for compatibility
        if labels[i].lower() == 'icnn':
            space.requires_grad = True
            pred_fx = grad(space)
            output = torch.autograd.grad(
                    outputs=pred_fx,
                    inputs=space,
                    grad_outputs=torch.ones_like(pred_fx),
                    create_graph=True
                )[0].detach()
            space.requires_grad = False # Reset
        else:
            output = grad(space).detach()

        error_map = (targ - output).norm(dim=1).reshape(Xi.shape).T
        error_maps.append(error_map)
        max_error = max(max_error, error_map.max().item())

    for i, (error_map, label) in enumerate(zip(error_maps, labels)):
        contour = ax[i].contourf(xi, yi, error_map, levels=50, cmap="RdYlGn_r", vmax=max_error)
        errors = error_map.flatten()
        rmse = torch.sqrt(torch.mean(errors ** 2))

        ax[i].set_title(f'{label} (RMSE={rmse:.2f})')
        ax[i].tick_params(axis='both', which='major')

    cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.9)
    plt.show()

def plot_distribution_comparison(
    model_output: np.ndarray, 
    expected_dist: np.ndarray, 
    model_name: str
):
    """
    Scatter plot comparing model output distribution vs expected distribution.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(model_output[:, 0], model_output[:, 1], color='#1f77b4', alpha=0.5, label='Model Output')
    plt.scatter(expected_dist[:, 0], expected_dist[:, 1], color='#ff7f0e', alpha=0.5, label='Expected Distribution')
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Comparison: {model_name} vs Expected")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
])

def plot_rgb_dist(input_image):
    # Convert the image to a NumPy array
    image_np = input_image.numpy()

    # Split the channels (R, G, B)
    red_channel = image_np[:, :, 0].flatten()  # Red
    green_channel = image_np[:, :, 1].flatten()  # Green
    blue_channel = image_np[:, :, 2].flatten()  # Blue

    # Plot histograms for each channel
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(red_channel, bins=256, color="red", alpha=0.7)
    plt.title("Red Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(green_channel, bins=256, color="green", alpha=0.7)
    plt.title("Green Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(blue_channel, bins=256, color="blue", alpha=0.7)
    plt.title("Blue Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    
def plot_model_test(model, image_path, target_image, save_path="output.png"):
    test_image = transform(Image.open(image_path))
    with torch.no_grad():
        result = model(test_image.permute(1, 2, 0).view(-1, 3))
    result = result.view(test_image.permute(1, 2, 0).shape).clip(0, 1)

    # Transport image
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    # Original image
    ax[0].imshow(test_image.permute(1, 2, 0).detach().cpu().numpy())
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Transported image
    ax[1].imshow(result.detach().cpu().numpy())
    ax[1].set_title('Transported Colors')
    ax[1].axis('off')

    # Target colors
    ax[2].imshow(target_image.permute(1, 2, 0).detach().cpu().numpy())
    ax[2].set_title('Target colors')
    ax[2].axis('off')

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
 
    plt.show()

def plot_compare_loss(train_loss: List[List[float]], train_cost: List[List[float]], labels: List[str]):
    """
    Compare training loss and cost across multiple models using Matplotlib.
    
    Args:
        train_loss (List[List[float]]): List of loss histories for each model.
        train_cost (List[List[float]]): List of cost histories for each model.
        labels (List[str]): Names of the models.
    """
    epochs = range(1, len(train_loss[0]) + 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Train Loss
    for i, loss in enumerate(train_loss):
        ax[0].plot(epochs, loss, label=labels[i])
    
    ax[0].set_title("Training Loss Comparison")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot Train Cost
    for i, cost in enumerate(train_cost):
        ax[1].plot(epochs, cost, label=labels[i])
    
    ax[1].set_title("Training Cost Comparison")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Cost")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def find_gm_components(target_image):
    # Convert the target image to a NumPy array (example tensor to simulate your setup)
    all_channels = target_image.permute(1, 2, 0).view(-1, 3).numpy()  # Assuming target_image is a torch tensor

    # Range of components to try
    components_range = range(1, 11)  # Testing GMMs with 1 to 10 components
    log_likelihoods = []  # To store log-likelihoods for each model

    for n_components in components_range:
        # Fit the GMM model
        gmm_model = GaussianMixture(n_components=n_components, random_state=42)
        gmm_model.fit(all_channels)
        
        # Store the log-likelihood
        log_likelihoods.append(gmm_model.lower_bound_)

    # Plot the log-likelihoods
    plt.figure(figsize=(8, 6))
    plt.plot(components_range, log_likelihoods, marker='o', linestyle='-', color='b')
    plt.title('Log-Likelihood vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plt.show()