import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from monotone_grad_nets.trainers.trainer import Trainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    def forward(self, x):
        return self.linear(x)

def test_scheduler_steplr():
    model = SimpleModel()
    # Dummy data
    x = torch.randn(10, 2)
    y = torch.randn(10, 2)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    initial_lr = 0.1
    trainer = Trainer(
        model=model,
        task='gradient',
        n_epochs=2,
        lr=initial_lr,
        scheduler='step',
        scheduler_kwargs={'step_size': 1, 'gamma': 0.1},
        verbose=False,
        criterion='mse'
    )
    
    trainer.train(loader)
    
    # Initial: 0.1
    # End of Epoch 1: 0.1 * 0.1 = 0.01
    # End of Epoch 2: 0.01 * 0.1 = 0.001
    final_lr = trainer.optimizer.param_groups[0]['lr']
    assert final_lr == pytest.approx(0.001)

def test_scheduler_invalid():
    model = SimpleModel()
    with pytest.raises(ValueError, match="Scheduler invalid not supported"):
        Trainer(model=model, scheduler='invalid')
