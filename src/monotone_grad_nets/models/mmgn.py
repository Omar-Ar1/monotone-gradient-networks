import torch
import torch.nn as nn
import torch.nn.init as init

class M_MGN(nn.Module):
    """
    Modular Monotone Gradient Network (M_MGN).

    This model employs custom structured layers and terms to ensure specific properties
    like monotonicity or convexity, utilizing a PSD term V^T V and log-cosh transformations.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of units in the hidden layers.
        num_layers (int): The number of custom layers in the network.
        W_k (nn.ModuleList): A list of linear layers (weights W_k) for each layer.
        activations (nn.ModuleList): A list of activation functions (e.g., Tanh) for each layer.
        V (nn.Linear): A learnable linear layer for the term V^T V, initialized to be orthogonal.
        a (nn.Parameter): A learned bias term added to the final output.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        """
        Initialize the M_MGN model.

        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Number of units in the hidden layers.
            num_layers (int): Number of custom layers.
        """
        super(M_MGN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define modules (W_k, b_k, and activation functions)
        self.W_k = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=True) for _ in range(num_layers)
        ])

        # Each module has its own activation (e.g., tanh, softplus)
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(num_layers)])

        # V^T V term (PSD by construction)
        self.V = nn.Linear(input_dim, input_dim, bias=False)  # Shape: [input_dim, input_dim]
        nn.init.orthogonal_(self.V.weight)  # Initialize V to be orthogonal

        # Bias term (a)
        self.a = nn.Parameter(torch.randn(input_dim))  # Learned bias


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the M_MGN.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, input_dim].
        """
        batch_size = x.shape[0]

        # Initialize output with bias term (broadcasted to batch)
        out = self.a.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, input_dim]

        # Add V^T V x term (ensures PSD Jacobian)
        V_sq = self.V.weight.t() @ self.V.weight  # Shape: [input_dim, input_dim]
        out = out + x @ V_sq  # Shape: [batch_size, input_dim]

        # Loop over modules and compute terms
        for k in range(self.num_layers):
            # Compute z_k = W_k x + b_k
            z_k = self.W_k[k](x)  # Shape: [batch_size, hidden_dim]

            # Compute s_k(z_k) = sum_i log(cosh(z_k_i)) (scalar per sample)
            s_k = torch.sum(torch.log(torch.cosh(z_k)), dim=1)  # Shape: [batch_size]

            # Compute activation σ_k(z_k)
            sigma_k = self.activations[k](z_k)  # Shape: [batch_size, hidden_dim]

            # Compute s_k(z_k) * W_k^T σ_k(z_k)
            W_k_T = self.W_k[k].weight.t()  # Shape: [input_dim, hidden_dim]
            term = (W_k_T @ sigma_k.t()).t()  # Shape: [batch_size, input_dim]
            term = s_k.unsqueeze(-1) * term  # Broadcast s_k and multiply

            out += term

        return out  # Shape: [batch_size, input_dim]

    def logcosh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute element-wise log(cosh(x)).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return torch.log(torch.cosh(x))
