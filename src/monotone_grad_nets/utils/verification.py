import torch
import numpy as np
from typing import Callable, Union

def verify_psd(func: Callable, model: torch.nn.Module, batch_size: int = 10000) -> str:
    """
    Verify if the Jacobian of the model is Positive Semi-Definite (PSD).
    
    Args:
        func (Callable): Function to generate input data.
        model (torch.nn.Module): The model to verify.
        batch_size (int): Number of samples to test.
        
    Returns:
        str: 'Jacobian is PSD' or 'Jacobian is not PSD'.
    """
    batch = func(torch.rand(size=(batch_size, 2)))
    input_dim = batch.shape[1]

    # Calculate the Jacobian for each sample in the batch
    # Note: This can be memory intensive for large batch_size
    jacobians = [torch.autograd.functional.jacobian(model, batch[i].unsqueeze(0)) for i in range(batch.shape[0])]

    # Reshape each Jacobian to be a square matrix
    jacobians = [j.reshape(input_dim, input_dim) for j in jacobians]

    # Check PSD for each Jacobian in the batch
    for jacobian in jacobians:
        # Check eigenvalues
        if not torch.all(torch.linalg.eigvalsh(jacobian) >= -1e-6): 
            return 'Jacobian is not PSD' 
    
    return 'Jacobian is PSD'

def get_gradient(x: torch.Tensor, pred_fx: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the output w.r.t. inputs.
    
    Args:
        x (torch.Tensor): Input tensor.
        pred_fx (torch.Tensor): Output tensor (scalar or vector).
        
    Returns:
        torch.Tensor: Gradient tensor.
    """
    # Ensure x requires gradient tracking
    if not x.requires_grad:
        x.requires_grad_()

    # Compute gradient of the output w.r.t. inputs
    return torch.autograd.grad(
        outputs=pred_fx,
        inputs=x,
        grad_outputs=torch.ones_like(pred_fx),
        create_graph=True
    )[0]