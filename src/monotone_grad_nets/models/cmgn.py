import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class C_MGN(nn.Module):
    """
    Cascade Monotone Gradient Network (C_MGN).

    This model employs a cascading structure with shared parameters across layers and 
    monotonic activation functions to compute output representations. It ensures flexible 
    and efficient processing of input data by utilizing shared weights, multiple bias terms, 
    and orthogonality constraints when required.

    Attributes:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Number of units in the hidden layers.
        output_dim (int): Dimensionality of the output data.
        num_layers (int): Number of cascading layers in the network.
        W (torch.nn.Parameter): Shared weight matrix [input_dim, hidden_dim].
        biases (torch.nn.ParameterList): List of bias vectors for each layer.
        bL (torch.nn.Parameter): Bias term added to the final output.
        V (torch.nn.Parameter): Weight matrix for additional transformations [input_dim, output_dim].
        activation (Callable): The activation function used in each layer.
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int, 
        ortho: bool = False, 
        activation: str = 'sigmoid'
    ):
        """
        Initialize the C_MGN model.

        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Number of units in the hidden layers.
            output_dim (int): Dimensionality of the output data.
            num_layers (int): Number of cascading layers.
            ortho (bool, optional): Whether to initialize matrix `V` as orthogonal. Defaults to False.
            activation (str, optional): Type of activation function ('sigmoid', 'tanh', 'softplus'). Defaults to 'sigmoid'.
        
        Raises:
            ValueError: If an unsupported activation function is specified.
        """
        super(C_MGN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Shared weight matrix across layers
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim))
        
        # Biases b0 to bL
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)])
        self.bL = nn.Parameter(torch.randn(output_dim))
        
        self.V = nn.Parameter(torch.randn(input_dim, output_dim))
        if ortho and input_dim == output_dim:
            nn.init.orthogonal_(self.V)  # Initialize V to be orthogonal

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the C_MGN.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        # First layer
        z_prev = torch.matmul(x, self.W) + self.biases[0]
        
        for l in range(1, self.num_layers):
            z_l = torch.matmul(x, self.W) + self.activation(z_prev) + self.biases[l]
            z_prev = z_l
            
        # (batch_size, hidden_dim) * (hidden_dim, input_dim) -> (batch_size, input_dim)
        inter_1 = torch.matmul(self.activation(z_prev), self.W.t()) 
        
        # x@V (b, i) * (i, o) => (b, o)
        # x@V@V.T (b, o) * (o, i) => (b, i)
        # Note: The original code comment said (b, i) but V is [input_dim, output_dim].
        # If input_dim == output_dim, then V.t() is [output_dim, input_dim].
        # x @ V is [batch, output]. (x @ V) @ V.t() is [batch, input].
        inter_2 = torch.matmul(torch.matmul(x, self.V), self.V.t()) 
        
        output = inter_1 + inter_2 + self.bL
        
        return output

