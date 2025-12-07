import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class I_CGN(nn.Module):
    """
    Input Convex Gradient Network (I_CGN).
    
    This model implements a gradient field of a convex function using a specific architecture
    involving line integrals of a potential field.
    
    Attributes:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Number of units in the hidden layers.
        output_dim (int): Dimensionality of the output, matching input_dim by default.
        lin (torch.nn.ModuleList): A list of linear layers for input and hidden transformations.
        act (torch.nn.Module): Activation function used across layers (Sigmoid).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, **kwargs):
        """
        Initialize the I_CGN model.

        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Number of units in the hidden layers.
            num_layers (int, optional): Number of hidden layers. Defaults to 1.
        """
        super(I_CGN, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = input_dim

        self.lin = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim, bias=True),
                *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
        ])

        self.act = nn.Sigmoid()

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian-Vector Product.
        
        Args:
            x (torch.Tensor): Input points.
            v (torch.Tensor): Tangent vectors.
            
        Returns:
            torch.Tensor: JVP result.
        """
        with torch.enable_grad():
            # computes w = V(x)v 
            w = torch.autograd.functional.jvp(self.forward_, inputs=x, v=v, create_graph=True)[1]
            # compute w = V.T(x)w
            w = torch.autograd.functional.vjp(self.forward_, inputs=x, v=w, create_graph=True)[1]
        return w

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal forward pass for the potential field.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output of the potential field network.
        """
        a = self.act
        y = a(self.lin[0](x))

        for lay in self.lin[1:-1]:
            y = a(lay(y))
        
        return self.lin[-1](y) if len(self.lin) > 1 else y
    
    def checkjac(self, simp: bool = False, N: int = 100):
        """
        Debug method to check Jacobian properties (PSD).
        
        Args:
            simp (bool): Use Simpson's rule if True.
            N (int): Number of integration points.
        """
        x = torch.rand(1, self.input_dim)
        print("computing jacobian using autograd on forward")
        if self.input_dim < 10:
            if simp:
                M = torch.autograd.functional.jacobian(self.forward_simp, inputs=x, vectorize=True).squeeze()
            else:
                M = torch.autograd.functional.jacobian(lambda x: self.forward(x, N=N), inputs=x, vectorize=True).squeeze()
            print(M)

        print("computing jacobian using autograd on forward_")
        if self.input_dim < 10:
            V = torch.autograd.functional.jacobian(self.forward_, inputs=x, vectorize=True).squeeze()
            V = V.T @ V 
            print(V)

        print("checking PSD")
        # M.detach() # This line was doing nothing effectively
        with torch.no_grad():
            for _ in range(100):
                v = torch.rand(self.input_dim)
                if (v * (M @ v)).sum() < -1e-2:
                    print("Failed PSD test", (v * (M @ v)).sum(), "vector", v, "at point", x)
                    return
        print("passed psd test")
        return 

    def forward(self, x: torch.Tensor, N: int = 100, **kwargs) -> torch.Tensor:
        """
        Forward pass computing the gradient via numerical integration.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim].
            N (int, optional): Number of integration points. Defaults to 100.
            
        Returns:
            torch.Tensor: Gradient output [batch_size, output_dim].
        """
        pts = torch.rand(size=(N,)).reshape(1, -1, 1).to(x.device)

        in_size = x.shape[0]
        input_dim = (self.input_dim,)
        output_dim = (self.output_dim,)

        z = x.unsqueeze(1) * pts
        v = x.unsqueeze(1) * torch.ones_like(pts)

        z = z.reshape(-1, *input_dim)
        v = v.reshape(-1, *input_dim)

        # this computes the integral
        y = self.jvp(z, v)
        y = y.reshape(in_size, -1, *output_dim).sum(dim=1) / N 
        return y

    def forward_simp(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass using Simpson's rule for integration.
        
        Args:
            x (torch.Tensor): Input tensor.
            w (torch.Tensor, optional): Optional weight tensor.
            
        Returns:
            torch.Tensor: Gradient output.
        """
        pts = torch.tensor([0, 1/3, 2/3, 1]).reshape(1, -1, 1).to(x.device)
        scale = torch.tensor([1, 3, 3, 1]).reshape(1, -1, 1).to(x.device)

        in_size = x.shape[0]
        input_dim = (self.input_dim,)
        output_dim = (self.output_dim,)

        z = x.unsqueeze(1) * pts
        if w is None: 
            v = x.unsqueeze(1) * torch.ones_like(pts)
        else: 
            v = w.unsqueeze(1) * torch.ones_like(pts)
            
        z = z.reshape(-1, *input_dim)
        v = v.reshape(-1, *input_dim)

        # this computes the integral
        y = self.jvp(z, v)
        y = 1/8 * (y.reshape(in_size, -1, *output_dim) * scale).sum(dim=1)

        return y