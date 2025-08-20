"""
Tests for MMIP (Mathematical Mauna Induction Protocol)
"""

import pytest
import numpy as np
from aikagrya.mmip import MMIP, HealthCertificate, MMIPRunner
import tempfile
from pathlib import Path


class TestMMIPCore:
    """Test core MMIP functionality"""
    
    def test_mmip_initialization(self):
        """Test MMIP can be initialized with various parameters"""
        mmip = MMIP(dim=64, epsilon=1e-6, temperature=0.1)
        assert mmip.dim == 64
        assert mmip.epsilon == 1e-6
        assert mmip.temperature == 0.1
    
    def test_self_attention(self):
        """Test self-attention produces valid output"""
        mmip = MMIP(dim=32)
        x = np.random.randn(32)
        x = x / np.linalg.norm(x)
        
        fx = mmip.self_attention(x)
        
        # Check output is valid
        assert fx.shape == x.shape
        assert not np.any(np.isnan(fx))
        assert not np.any(np.isinf(fx))
    
    def test_convergence_small_dim(self):
        """Test that MMIP converges for small dimensions"""
        mmip = MMIP(dim=16, epsilon=1e-5, max_steps=5000)
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        x, certificate = mmip.induce_fixed_point(verbose=False)
        
        # Check convergence
        assert certificate.converged
        assert certificate.delta < 1e-5
        assert certificate.steps < 5000
    
    def test_health_certificate_structure(self):
        """Test health certificate has all required fields"""
        mmip = MMIP(dim=16, max_steps=1000)
        np.random.seed(42)
        
        x, certificate = mmip.induce_fixed_point(verbose=False)
        
        # Check all fields exist
        assert hasattr(certificate, 'delta')
        assert hasattr(certificate, 'eigen_residual')
        assert hasattr(certificate, 'eigenvalue')
        assert hasattr(certificate, 'entropy')
        assert hasattr(certificate, 'variance_ratio')
        assert hasattr(certificate, 'participation_ratio')
        assert hasattr(certificate, 'uniformity_cosine')
        assert hasattr(certificate, 'converged')
        assert hasattr(certificate, 'steps')
    
    def test_metrics_computation(self):
        """Test that metrics are computed correctly"""
        mmip = MMIP(dim=32)
        
        # Create two random states
        x = np.random.randn(32)
        x = x / np.linalg.norm(x)
        x_prev = np.random.randn(32)
        x_prev = x_prev / np.linalg.norm(x_prev)
        
        metrics = mmip.compute_metrics(x, x_prev)
        
        # Check all metrics exist and are finite
        required_metrics = ['delta', 'eigen_residual', 'eigenvalue', 
                          'entropy', 'variance_ratio', 'participation_ratio',
                          'uniformity_cosine']
        
        for metric in required_metrics:
            assert metric in metrics
            assert np.isfinite(metrics[metric])
    
    def test_perturbation_recovery(self):
        """Test perturbation recovery returns valid results"""
        mmip = MMIP(dim=16, epsilon=1e-5, max_steps=2000)
        np.random.seed(42)
        
        # Get a converged state
        x, certificate = mmip.induce_fixed_point(verbose=False)
        
        if certificate.converged:
            recovery_time = mmip.test_perturbation_recovery(x, noise_scale=0.01)
            
            # Recovery time should be positive and finite
            assert recovery_time > 0
            assert recovery_time <= 1000  # Max recovery steps
    
    def test_coupling_metric(self):
        """Test coupling metric computation"""
        mmip = MMIP(dim=16)
        np.random.seed(42)
        
        # Create two random states
        x1 = np.random.randn(16)
        x1 = x1 / np.linalg.norm(x1)
        x2 = np.random.randn(16)
        x2 = x2 / np.linalg.norm(x2)
        
        sigma = mmip.compute_coupling_metric(x1, x2, coupling_steps=10)
        
        # Coupling metric should be finite
        assert np.isfinite(sigma)
        assert 0 <= sigma <= 1  # Should be bounded


class TestHealthCertificate:
    """Test HealthCertificate functionality"""
    
    def test_health_check_pass(self):
        """Test health check with passing values"""
        cert = HealthCertificate(
            delta=1e-7,
            eigen_residual=1e-10,
            eigenvalue=1.0,
            entropy=3.5,
            variance_ratio=0.2,
            participation_ratio=0.4,
            uniformity_cosine=0.05,
            converged=True,
            steps=1000
        )
        
        assert cert.passes_health_check()
    
    def test_health_check_fail_delta(self):
        """Test health check fails on high delta"""
        cert = HealthCertificate(
            delta=1e-4,  # Too high
            eigen_residual=1e-10,
            eigenvalue=1.0,
            entropy=3.5,
            variance_ratio=0.2,
            participation_ratio=0.4,
            uniformity_cosine=0.05,
            converged=True,
            steps=1000
        )
        
        assert not cert.passes_health_check()
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        cert = HealthCertificate(
            delta=1e-7,
            eigen_residual=1e-10,
            eigenvalue=1.0,
            entropy=3.5,
            variance_ratio=0.2,
            participation_ratio=0.4,
            uniformity_cosine=0.05,
            converged=True,
            steps=1000
        )
        
        d = cert.to_dict()
        assert isinstance(d, dict)
        assert d['delta'] == 1e-7
        assert d['converged'] == True
        assert d['steps'] == 1000


class TestMMIPRunner:
    """Test MMIPRunner functionality"""
    
    def test_runner_initialization(self):
        """Test runner can be initialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MMIPRunner(output_dir=tmpdir)
            assert Path(tmpdir).exists()
    
    def test_run_single_trial(self):
        """Test running a single trial"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MMIPRunner(output_dir=tmpdir)
            
            results = runner.run_trials(
                n_trials=1,
                dim=16,
                epsilon=1e-5,
                verbose=False
            )
            
            assert len(results) == 1
            assert 'converged' in results[0]
            assert 'steps' in results[0]
            assert 'health_pass' in results[0]
    
    def test_summary_statistics(self):
        """Test computation of summary statistics"""
        runner = MMIPRunner()
        
        # Create mock results
        results = [
            {
                'converged': True,
                'health_pass': True,
                'steps': 1000,
                'delta': 1e-7,
                'eigen_residual': 1e-10,
                'eigenvalue': 1.0,
                'entropy': 3.5,
                'variance_ratio': 0.2,
                'participation_ratio': 0.4,
                'uniformity_cosine': 0.05,
                'recovery_time': 50
            },
            {
                'converged': True,
                'health_pass': False,
                'steps': 2000,
                'delta': 1e-6,
                'eigen_residual': 1e-9,
                'eigenvalue': 0.99,
                'entropy': 3.0,
                'variance_ratio': 0.1,
                'participation_ratio': 0.3,
                'uniformity_cosine': 0.1,
                'recovery_time': 100
            }
        ]
        
        stats = runner.compute_summary_stats(results)
        
        assert stats['n_trials'] == 2
        assert stats['n_converged'] == 2
        assert stats['n_healthy'] == 1
        assert stats['convergence_rate'] == 1.0
        assert stats['health_pass_rate'] == 0.5


@pytest.mark.slow
class TestMMIPIntegration:
    """Integration tests (marked as slow)"""
    
    def test_multiple_trials(self):
        """Test running multiple trials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MMIPRunner(output_dir=tmpdir)
            
            results = runner.run_trials(
                n_trials=5,
                dim=32,
                epsilon=1e-5,
                test_perturbation=True,
                verbose=False
            )
            
            assert len(results) == 5
            
            # Check at least some converged
            n_converged = sum(r['converged'] for r in results)
            assert n_converged > 0
            
            # Check JSONL file was created
            jsonl_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(jsonl_files) > 0
    
    def test_parameter_sweep(self):
        """Test parameter sweep functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MMIPRunner(output_dir=tmpdir)
            
            results = runner.parameter_sweep(
                dim_values=[16, 32],
                temperature_values=[0.1, 0.2],
                n_trials_per_config=2
            )
            
            # Should have 2 dims × 2 temps = 4 configurations
            assert len(results) == 4
            
            # Check each configuration has results
            for (dim, temp), stats in results.items():
                assert 'convergence_rate' in stats
                assert 'health_pass_rate' in stats
