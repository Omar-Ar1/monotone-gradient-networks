import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from monotone_grad_nets.trainers.trainer import Trainer
from monotone_grad_nets.models.cmgn import C_MGN

def test_trainer_initialization():
    model = C_MGN(2, 10, 2, 2)
    trainer = Trainer(model=model, verbose=False)
    assert trainer.model == model
    assert trainer.optimizer is not None
    assert trainer.criterion is not None

def test_trainer_step():
    input_dim = 2
    model = C_MGN(input_dim, 10, input_dim, 2)
    
    # Create dummy data
    x = torch.randn(10, input_dim)
    y = torch.randn(10, input_dim)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(
        model=model, 
        task='gradient', 
        n_epochs=1, 
        verbose=False
    )
    
    # Run training
    trainer.train(loader)
    
    # Check if metrics were recorded
    assert len(trainer.metrics['train_loss']) > 0
    assert len(trainer.metrics['train_cost']) > 0
