
import torch
import torch.nn as nn
import pytest
import time
from src.monotone_grad_nets.trainers.trainer import Trainer
from src.monotone_grad_nets.models.mmgn import M_MGN

class TestTrainerOptimization:
    """Test optimized Jacobian computation in Trainer."""
    
    def test_jacobian_correctness(self):
        """Verify that vectorized Jacobian matches loop-based Jacobian."""
        d = 10
        batch_size = 32
        
        # Create model and trainer
        model = M_MGN(input_dim=d, hidden_dim=20, num_layers=2)
        trainer = Trainer(
            model=model,
            task='optimal_transport',
            device='cpu' # Test on CPU first
        )
        
        x = torch.randn(batch_size, d)
        
        # Compute using optimized method
        # Note: We can't easily access the old loop method unless we mock TORCH_FUNC_AVAILABLE to False
        # or manually implement the loop here to compare.
        
        # Optimized result
        jac_optimized = trainer._compute_batch_jacobian(x)
        
        # Manual loop result
        jac_loop = torch.zeros(batch_size, d, d)
        for i in range(batch_size):
            jac_loop[i] = torch.autograd.functional.jacobian(
                lambda x_in: trainer._get_grad(x_in.unsqueeze(0)).squeeze(0),
                x[i]
            )
            
        # Compare
        diff = (jac_optimized - jac_loop).abs().max()
        assert torch.allclose(jac_optimized, jac_loop, atol=1e-5), \
            f"Jacobian mismatch. Max diff: {diff.item()}"
            
    def test_performance_improvement(self):
        """Benchmark performance improvement."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping performance test on CPU (might not be faster due to overhead)")
            
        d = 50
        batch_size = 64
        device = 'cuda'
        
        model = M_MGN(input_dim=d, hidden_dim=50, num_layers=2).to(device)
        trainer = Trainer(model=model, task='optimal_transport', device=device)
        
        x = torch.randn(batch_size, d, device=device)
        
        # Warmup
        _ = trainer._compute_batch_jacobian(x)
        
        # Measure optimized time
        start = time.time()
        for _ in range(10):
            _ = trainer._compute_batch_jacobian(x)
        torch.cuda.synchronize()
        time_optimized = (time.time() - start) / 10
        
        # Measure loop time (simulate loop)
        start = time.time()
        # Run only once because it's slow
        jac_loop = torch.zeros(batch_size, d, d, device=device)
        for i in range(batch_size):
            jac_loop[i] = torch.autograd.functional.jacobian(
                lambda x_in: trainer._get_grad(x_in.unsqueeze(0)).squeeze(0),
                x[i]
            )
        torch.cuda.synchronize()
        time_loop = time.time() - start
        
        print(f"\nAvg Optimized Time: {time_optimized:.4f}s")
        print(f"Loop Time (1 run): {time_loop:.4f}s")
        print(f"Speedup: {time_loop / time_optimized:.2f}x")
        
        # Expect significant speedup (e.g. > 2x)
        assert time_loop > time_optimized * 2, "Expected at least 2x speedup"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
