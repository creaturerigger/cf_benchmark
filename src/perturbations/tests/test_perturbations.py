import pytest
import torch

from src.perturbations.gaussian import GaussianPerturbation
from src.perturbations.uniform import UniformPerturbation


# ── GaussianPerturbation ─────────────────────────────────────


class TestGaussianInit:
    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            GaussianPerturbation(sigma=-0.1)

    def test_zero_sigma_no_change(self):
        p = GaussianPerturbation(sigma=0.0)
        x = torch.tensor([0.5, 0.5])
        assert torch.equal(p(x), x)


class TestGaussianCall:
    def test_output_shape_1d(self):
        p = GaussianPerturbation(sigma=0.1)
        x = torch.tensor([0.5, 0.3, 0.7])
        assert p(x).shape == x.shape

    def test_output_shape_2d(self):
        p = GaussianPerturbation(sigma=0.1)
        x = torch.tensor([[0.5, 0.3, 0.7]])
        assert p(x).shape == x.shape

    def test_clamped_to_01(self):
        p = GaussianPerturbation(sigma=10.0)  # large sigma to force clamping
        x = torch.tensor([0.5, 0.5, 0.5])
        for _ in range(20):
            x_prime = p(x)
            assert (x_prime >= 0.0).all()
            assert (x_prime <= 1.0).all()

    def test_does_not_modify_original(self):
        p = GaussianPerturbation(sigma=0.1)
        x = torch.tensor([0.5, 0.5])
        x_copy = x.clone()
        _ = p(x)
        assert torch.equal(x, x_copy)

    def test_continuous_indices_only_perturb_selected(self):
        p = GaussianPerturbation(sigma=10.0, continuous_indices=[0])
        x = torch.tensor([0.5, 1.0, 0.0, 0.0])  # col 0 cont, rest cat
        x_prime = p(x)
        # Categorical columns should be untouched
        assert x_prime[1] == 1.0
        assert x_prime[2] == 0.0
        assert x_prime[3] == 0.0

    def test_deterministic_with_seed(self):
        p = GaussianPerturbation(sigma=0.1)
        x = torch.tensor([0.5, 0.5])
        torch.manual_seed(0)
        a = p(x)
        torch.manual_seed(0)
        b = p(x)
        assert torch.equal(a, b)


class TestGaussianGenerate:
    def test_returns_correct_count(self):
        p = GaussianPerturbation(sigma=0.1)
        x = torch.tensor([0.5, 0.5])
        results = p.generate(x, m=10)
        assert len(results) == 10

    def test_each_has_correct_shape(self):
        p = GaussianPerturbation(sigma=0.1)
        x = torch.tensor([[0.5, 0.3]])
        results = p.generate(x, m=5)
        for r in results:
            assert r.shape == x.shape

    def test_results_differ(self):
        p = GaussianPerturbation(sigma=0.5)
        x = torch.tensor([0.5, 0.5, 0.5])
        results = p.generate(x, m=10)
        unique = {tuple(r.tolist()) for r in results}
        assert len(unique) > 1


# ── UniformPerturbation ──────────────────────────────────────


class TestUniformInit:
    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon must be >= 0"):
            UniformPerturbation(epsilon=-0.1)

    def test_zero_epsilon_no_change(self):
        p = UniformPerturbation(epsilon=0.0)
        x = torch.tensor([0.5, 0.5])
        assert torch.equal(p(x), x)


class TestUniformCall:
    def test_output_shape_1d(self):
        p = UniformPerturbation(epsilon=0.1)
        x = torch.tensor([0.5, 0.3, 0.7])
        assert p(x).shape == x.shape

    def test_output_shape_2d(self):
        p = UniformPerturbation(epsilon=0.1)
        x = torch.tensor([[0.5, 0.3, 0.7]])
        assert p(x).shape == x.shape

    def test_clamped_to_01(self):
        p = UniformPerturbation(epsilon=10.0)
        x = torch.tensor([0.5, 0.5, 0.5])
        for _ in range(20):
            x_prime = p(x)
            assert (x_prime >= 0.0).all()
            assert (x_prime <= 1.0).all()

    def test_does_not_modify_original(self):
        p = UniformPerturbation(epsilon=0.1)
        x = torch.tensor([0.5, 0.5])
        x_copy = x.clone()
        _ = p(x)
        assert torch.equal(x, x_copy)

    def test_bounded_perturbation(self):
        """With small epsilon and values away from edges, noise stays in [-ε, ε]."""
        eps = 0.05
        p = UniformPerturbation(epsilon=eps)
        x = torch.full((100,), 0.5)
        for _ in range(20):
            x_prime = p(x)
            diff = (x_prime - x).abs()
            assert (diff <= eps + 1e-7).all()

    def test_continuous_indices_only_perturb_selected(self):
        p = UniformPerturbation(epsilon=10.0, continuous_indices=[0])
        x = torch.tensor([0.5, 1.0, 0.0, 0.0])
        x_prime = p(x)
        assert x_prime[1] == 1.0
        assert x_prime[2] == 0.0
        assert x_prime[3] == 0.0

    def test_deterministic_with_seed(self):
        p = UniformPerturbation(epsilon=0.1)
        x = torch.tensor([0.5, 0.5])
        torch.manual_seed(0)
        a = p(x)
        torch.manual_seed(0)
        b = p(x)
        assert torch.equal(a, b)


class TestUniformGenerate:
    def test_returns_correct_count(self):
        p = UniformPerturbation(epsilon=0.1)
        x = torch.tensor([0.5, 0.5])
        results = p.generate(x, m=10)
        assert len(results) == 10

    def test_results_differ(self):
        p = UniformPerturbation(epsilon=0.5)
        x = torch.tensor([0.5, 0.5, 0.5])
        results = p.generate(x, m=10)
        unique = {tuple(r.tolist()) for r in results}
        assert len(unique) > 1
