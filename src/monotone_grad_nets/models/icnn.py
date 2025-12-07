import torch
import torch.nn as nn
import torch.nn.init as init

class I_CNN(nn.Module):
    """
    Input Convex Neural Network (I_CNN).

    This model ensures convexity with respect to its input by constraining certain weights 
    to be non-negative and applying ReLU activations. It features a modular architecture where 
    each layer uses shared components for flexibility and scalability.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of units in the hidden layers.
        num_layers (int): The number of layers in the network.
        relu (torch.nn.ReLU): ReLU activation function for introducing non-linearity.
        modules_x (torch.nn.ModuleList): A list of `Linear` layers for transformations on the input data.
        modules_z (torch.nn.ModuleList): A list of `Linear` layers for transformations on intermediate outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        """
        Initialize the I_CNN model.

        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Number of units in the hidden layers.
            num_layers (int): Number of layers in the network.
        """
        super(I_CNN, self).__init__()
        # Define ReLU activation
        self.relu = nn.ReLU()

        # Use nn.ModuleList for layers to ensure they are registered as model parameters
        self.modules_x = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=True) for _ in range(num_layers)])
        self.modules_z = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the I_CNN.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Final output tensor of shape [batch_size, hidden_dim].
        """
        # Initialize zl as a zero tensor with the same device as x
        zl = torch.zeros(x.shape[0], self.modules_x[0].out_features, device=x.device)
        for i in range(len(self.modules_x)):
            self.modules_z[i].weight.data.clamp_(min=0)
            zl = self.relu(self.modules_z[i](zl) + self.modules_x[i](x))  # Apply ReLU activation
        return zl

    def _initialize_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        """
        # Loop through all layers and apply Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Apply Xavier uniform initialization
                if module.bias is not None:
                    init.zeros_(module.bias)  # Initialize biases to 0