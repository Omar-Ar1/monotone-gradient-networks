import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from monotone_grad_nets.trainers.trainer import Trainer

# Simple linear model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def simple_data():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10)
    return loader

def test_trainer_init():
    model = SimpleModel()
    trainer = Trainer(model=model)
    assert trainer.model == model
    assert trainer.accumulation_steps == 1
    assert trainer.use_amp == False
    assert trainer.grad_clip_max_norm is None

def test_gradient_accumulation(simple_data):
    model = SimpleModel()
    # Use SGD to make gradient checking easier (no momentum state initially)
    trainer = Trainer(model=model, optimizer='SGD', accumulation_steps=2, n_epochs=1, verbose=False)
    
    # Mock optimizer step to count calls
    original_step = trainer.optimizer.step
    step_count = 0
    def mock_step():
        nonlocal step_count
        step_count += 1
        original_step()
    
    trainer.optimizer.step = mock_step
    
    trainer.train(simple_data)
    
    # 100 samples, batch size 10 -> 10 batches.
    # Accumulation steps 2 -> step every 2 batches.
    # Total steps = 10 / 2 = 5.
    assert step_count == 5

def test_gradient_clipping(simple_data):
    model = SimpleModel()
    # Set max norm very low to force clipping
    trainer = Trainer(model=model, optimizer='SGD', grad_clip_max_norm=0.001, n_epochs=1, verbose=False)
    
    # We can't easily check internal clipping without mocking torch.nn.utils.clip_grad_norm_
    # But we can run it and ensure no errors, and that grads are small if we inspected them.
    # For now, just ensure it runs.
    trainer.train(simple_data)

def test_nan_handling():
    model = SimpleModel()
    trainer = Trainer(model=model, n_epochs=1, verbose=False)
    
    # Create data that produces NaN loss (e.g., target is Inf)
    x = torch.randn(10, 10)
    y = torch.full((10, 1), float('inf'))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    # Should not crash, but print warning
    trainer.train(loader)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_amp(simple_data):
    model = SimpleModel()
    trainer = Trainer(model=model, use_amp=True, device='cuda', n_epochs=1, verbose=False)
    trainer.train(simple_data)
    assert trainer.scaler is not None
