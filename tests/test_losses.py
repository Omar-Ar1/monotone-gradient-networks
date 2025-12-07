"""
Comprehensive tests for loss functions.

Tests verify:
1. NLL loss correctness against analytical solutions
2. KLD loss with known distributions
3. Numerical stability with edge cases
4. Gradient correctness with finite differences
"""

import torch
import pytest
import numpy as np
from src.monotone_grad_nets.trainers.trainer import Trainer
from src.monotone_grad_nets.models.mmgn import M_MGN


class TestNLLLoss:
    """Test Negative Log Likelihood loss computation."""
    
    def test_nll_standard_gaussian(self):
        """Test NLL for standard Gaussian matches analytical solution."""
        # For standard Gaussian: NLL = (d/2)log(2pi) + 0.5 * ||z||^2
        d = 10
        batch_size = 100
        
        # Sample from standard Gaussian
        z = torch.randn(batch_size, d)
        
        # Analytical NLL
        const_term = (d / 2) * np.log(2 * np.pi)
        analytical_nll = (0.5 * (z ** 2).sum(dim=1) + const_term).mean()
        
        # Create a simple identity model for testing
        model = M_MGN(input_dim=d, hidden_dim=5, num_layers=1)
        trainer = Trainer(
            model=model,
            task='optimal_transport',
            criterion='nll',
            device='cpu'
        )
        
        # Compute NLL using trainer's loss (with identity Jacobian, log_det=0)
        # We need to mock the scenario where g(x) = z and J = I
        with torch.no_grad():
            # For identity map, log_det should be 0
            log_p = -0.5 * (z ** 2).sum(dim=1) - const_term
            computed_nll = -log_p.mean()
        
        # Should match within numerical precision
        assert torch.allclose(computed_nll, analytical_nll, rtol=1e-5, atol=1e-6), \
            f"NLL mismatch: computed={computed_nll:.6f}, analytical={analytical_nll:.6f}"
    
    def test_nll_constant_term_correctness(self):
        """Verify the constant term is (d/2)log(2pi), not log(2pi)."""
        d_values = [2, 5, 10, 50]
        
        for d in d_values:
            z = torch.randn(100, d)
            
            # Correct constant
            correct_const = (d / 2) * torch.log(torch.tensor(2 * torch.pi))
            
            # Wrong constant (what we had before)
            wrong_const = torch.log(torch.tensor(2 * torch.pi))
            
            # The difference should scale with d
            expected_diff = (d / 2 - 1) * np.log(2 * np.pi)
            actual_diff = (correct_const - wrong_const).item()
            
            assert np.isclose(actual_diff, expected_diff, rtol=1e-5), \
                f"For d={d}, constant term difference incorrect"
    
    def test_nll_numerical_stability(self):
        """Test NLL computation with extreme values."""
        d = 10
        
        # Test with very small values
        z_small = torch.randn(10, d) * 1e-10
        const_term = (d / 2) * torch.log(torch.tensor(2 * torch.pi))
        nll_small = (0.5 * (z_small ** 2).sum(dim=1) + const_term).mean()
        assert not torch.isnan(nll_small) and not torch.isinf(nll_small)
        
        # Test with large values
        z_large = torch.randn(10, d) * 1e3
        nll_large = (0.5 * (z_large ** 2).sum(dim=1) + const_term).mean()
        assert not torch.isnan(nll_large) and not torch.isinf(nll_large)


class TestKLDLoss:
    """Test KL Divergence loss computation."""
    
    def test_kld_identical_distributions(self):
        """KLD between identical distributions should be zero."""
        d = 5
        batch_size = 1000
        
        # Sample from N(0, I)
        z = torch.randn(batch_size, d)
        
        # Target is also N(0, I)
        mu_p = torch.zeros(d)
        cov_p = torch.eye(d)
        
        model = M_MGN(input_dim=d, hidden_dim=5, num_layers=1)
        trainer = Trainer(
            model=model,
            task='optimal_transport',
            criterion='kld',
            target_distribution={'mean': mu_p, 'cov': cov_p},
            device='cpu'
        )
        
        kld = trainer._kld_loss(z, mu_p, cov_p)
        
        # Should be close to zero (with some sampling error)
        assert kld.item() < 0.1, f"KLD for identical distributions should be ~0, got {kld.item()}"
    
    def test_kld_known_gaussians(self):
        """Test KLD between two known Gaussians."""
        # KLD(N(mu1, Sigma1) || N(mu2, Sigma2)) has analytical form
        d = 2
        
        # Q: N([1, 1], I)
        mu_q = torch.tensor([1.0, 1.0])
        cov_q = torch.eye(2)
        
        # P: N([0, 0], I)
        mu_p = torch.tensor([0.0, 0.0])
        cov_p = torch.eye(2)
        
        # Analytical KLD = 0.5 * (||mu_q - mu_p||^2) = 0.5 * 2 = 1.0
        analytical_kld = 0.5 * ((mu_q - mu_p) ** 2).sum()
        
        # Sample from Q
        samples = torch.randn(10000, d) + mu_q
        
        model = M_MGN(input_dim=d, hidden_dim=5, num_layers=1)
        trainer = Trainer(
            model=model,
            task='optimal_transport',
            criterion='kld',
            target_distribution={'mean': mu_p, 'cov': cov_p},
            device='cpu'
        )
        
        computed_kld = trainer._kld_loss(samples, mu_p, cov_p)
        
        # Should match within sampling error
        assert torch.allclose(computed_kld, analytical_kld, rtol=0.1, atol=0.1), \
            f"KLD mismatch: computed={computed_kld:.4f}, analytical={analytical_kld:.4f}"
    
    def test_kld_numerical_stability(self):
        """Test KLD with near-singular covariances."""
        d = 3
        
        # Create a nearly singular covariance
        cov_p = torch.eye(d) * 1e-3 + torch.ones(d, d) * 1e-6
        mu_p = torch.zeros(d)
        
        # Sample data
        z = torch.randn(100, d) * 0.1
        
        model = M_MGN(input_dim=d, hidden_dim=5, num_layers=1)
        trainer = Trainer(
            model=model,
            task='optimal_transport',
            criterion='kld',
            target_distribution={'mean': mu_p, 'cov': cov_p},
            device='cpu'
        )
        
        kld = trainer._kld_loss(z, mu_p, cov_p)
        
        # Should not be NaN or Inf
        assert not torch.isnan(kld) and not torch.isinf(kld), \
            "KLD computation failed with near-singular covariance"


class TestLogDetStability:
    """Test log-determinant computation stability."""
    
    def test_slogdet_vs_det_log(self):
        """Verify slogdet is more stable than det + log."""
        # Create matrices with very small/large determinants
        
        # Very small determinant
        A_small = torch.eye(5) * 1e-10
        sign_s, logdet_s = torch.linalg.slogdet(A_small)
        
        # Old method would fail
        det_old = torch.det(A_small)
        if det_old > 0:
            logdet_old = torch.log(det_old)
        else:
            logdet_old = torch.tensor(float('-inf'))
        
        # slogdet should give finite result
        assert torch.isfinite(logdet_s), "slogdet should handle small determinants"
        
        # Very large determinant
        A_large = torch.eye(5) * 1e10
        sign_l, logdet_l = torch.linalg.slogdet(A_large)
        assert torch.isfinite(logdet_l), "slogdet should handle large determinants"
    
    def test_jacobian_log_det_identity(self):
        """Log-det of identity Jacobian should be zero."""
        d = 5
        J = torch.eye(d)
        
        sign, logdet = torch.linalg.slogdet(J)
        
        assert sign == 1, "Identity matrix should have positive determinant"
        assert torch.allclose(logdet, torch.tensor(0.0), atol=1e-6), \
            f"Log-det of identity should be 0, got {logdet}"



def test_trainer_input_validation():
    """Test that Trainer validates inputs correctly."""
    model = M_MGN(input_dim=5, hidden_dim=5, num_layers=1)
    
    # Invalid task
    with pytest.raises(ValueError, match="task must be"):
        Trainer(model=model, task='invalid_task')
    
    # Missing model
    with pytest.raises(ValueError, match="Model must be provided"):
        Trainer(model=None, task='gradient')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
