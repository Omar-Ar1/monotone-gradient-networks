import pytest
import torch
from monotone_grad_nets.models import C_MGN, I_CGN, I_CNN, M_MGN

@pytest.mark.parametrize("input_dim, hidden_dim, output_dim, num_layers", [
    (2, 10, 2, 2),
    (10, 20, 10, 3)
])
def test_cmgn_shape(input_dim, hidden_dim, output_dim, num_layers):
    model = C_MGN(input_dim, hidden_dim, output_dim, num_layers)
    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5, output_dim)

@pytest.mark.parametrize("input_dim, hidden_dim, num_layers", [
    (2, 10, 2),
    (5, 20, 3)
])
def test_icgn_shape(input_dim, hidden_dim, num_layers):
    model = I_CGN(input_dim, hidden_dim, num_layers)
    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5, input_dim) # Output dim matches input dim for ICGN

@pytest.mark.parametrize("input_dim, hidden_dim, num_layers", [
    (2, 10, 2),
    (5, 20, 3)
])
def test_icnn_shape(input_dim, hidden_dim, num_layers):
    model = I_CNN(input_dim, hidden_dim, num_layers)
    x = torch.randn(5, input_dim)
    output = model(x)
    # ICNN output shape depends on the last layer, which is hidden_dim in the current implementation
    assert output.shape == (5, hidden_dim) 

@pytest.mark.parametrize("input_dim, hidden_dim, num_layers", [
    (2, 10, 2),
    (5, 20, 3)
])
def test_mmgn_shape(input_dim, hidden_dim, num_layers):
    model = M_MGN(input_dim, hidden_dim, num_layers)
    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5, input_dim)
