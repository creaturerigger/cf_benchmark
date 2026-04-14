import pytest
import torch

from src.perturbations.gaussian import GaussianPerturbation
from src.perturbations.base_perturbation import BasePerturbation
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


# ── Categorical perturbation (shared helper) ─────────────────

_resample = BasePerturbation.resample_categorical_groups


class TestResampleCategoricalGroups:
    """Tests for the resample_categorical_groups static method."""

    def test_output_is_valid_one_hot(self):
        """Result must still be a valid one-hot encoding per group."""
        groups = [[2, 3, 4]]  # 3-category feature at indices 2-4
        x = torch.tensor([0.5, 0.3, 1.0, 0.0, 0.0])
        torch.manual_seed(0)
        x_prime = _resample(x.clone(), groups, prob=1.0)
        cat_vals = x_prime[2:5]
        assert cat_vals.sum().item() == 1.0
        assert set(cat_vals.tolist()).issubset({0.0, 1.0})

    def test_prob_zero_no_change(self):
        groups = [[2, 3, 4]]
        x = torch.tensor([0.5, 0.3, 1.0, 0.0, 0.0])
        x_prime = _resample(x.clone(), groups, prob=0.0)
        assert torch.equal(x_prime, x)

    def test_prob_one_always_resamples(self):
        """With prob=1.0, at least some calls should change the category."""
        groups = [[2, 3, 4]]
        x = torch.tensor([0.5, 0.3, 1.0, 0.0, 0.0])
        changed = False
        for _ in range(50):
            x_prime = _resample(x.clone(), groups, prob=1.0)
            if not torch.equal(x_prime[2:5], x[2:5]):
                changed = True
                break
        assert changed

    def test_continuous_columns_untouched(self):
        """Only categorical group indices should be modified."""
        groups = [[2, 3, 4]]
        x = torch.tensor([0.5, 0.3, 1.0, 0.0, 0.0])
        x_prime = _resample(x.clone(), groups, prob=1.0)
        assert x_prime[0] == 0.5
        assert x_prime[1] == 0.3

    def test_multiple_groups(self):
        """Multiple categorical groups are resampled independently."""
        groups = [[2, 3], [4, 5, 6]]  # binary + 3-category
        x = torch.tensor([0.5, 0.3, 1.0, 0.0, 0.0, 1.0, 0.0])
        x_prime = _resample(x.clone(), groups, prob=1.0)
        # Group 1 valid
        g1 = x_prime[2:4]
        assert g1.sum().item() == 1.0
        assert set(g1.tolist()).issubset({0.0, 1.0})
        # Group 2 valid
        g2 = x_prime[4:7]
        assert g2.sum().item() == 1.0
        assert set(g2.tolist()).issubset({0.0, 1.0})

    def test_single_category_group_unchanged(self):
        """A group with only 1 category should never change."""
        groups = [[2]]
        x = torch.tensor([0.5, 0.3, 1.0])
        x_prime = _resample(x.clone(), groups, prob=1.0)
        assert x_prime[2] == 1.0


# ── Gaussian with categorical groups ─────────────────────────


class TestGaussianCategorical:
    def test_cat_groups_resampled(self):
        """Gaussian with cat groups should perturb both continuous and categorical."""
        groups = [[1, 2, 3]]
        p = GaussianPerturbation(sigma=0.1, continuous_indices=[0],
                                 categorical_groups=groups, cat_prob=1.0)
        x = torch.tensor([0.5, 1.0, 0.0, 0.0])
        changed = False
        for _ in range(50):
            x_prime = p(x)
            cat = x_prime[1:4]
            assert cat.sum().item() == pytest.approx(1.0)
            if not torch.equal(cat, x[1:4]):
                changed = True
                break
        assert changed

    def test_cat_prob_defaults_to_sigma(self):
        p = GaussianPerturbation(sigma=0.3)
        assert p.cat_prob == pytest.approx(0.3)

    def test_cat_prob_clamped(self):
        p = GaussianPerturbation(sigma=5.0)
        assert p.cat_prob == 1.0

    def test_no_cat_groups_backward_compat(self):
        """Without cat groups, behaviour is identical to before."""
        p = GaussianPerturbation(sigma=0.1, continuous_indices=[0])
        x = torch.tensor([0.5, 1.0, 0.0, 0.0])
        x_prime = p(x)
        # Categorical columns unchanged (no groups provided)
        assert x_prime[1] == 1.0
        assert x_prime[2] == 0.0
        assert x_prime[3] == 0.0


# ── Uniform with categorical groups ──────────────────────────


class TestUniformCategorical:
    def test_cat_groups_resampled(self):
        groups = [[1, 2, 3]]
        p = UniformPerturbation(epsilon=0.1, continuous_indices=[0],
                                categorical_groups=groups, cat_prob=1.0)
        x = torch.tensor([0.5, 1.0, 0.0, 0.0])
        changed = False
        for _ in range(50):
            x_prime = p(x)
            cat = x_prime[1:4]
            assert cat.sum().item() == pytest.approx(1.0)
            if not torch.equal(cat, x[1:4]):
                changed = True
                break
        assert changed

    def test_cat_prob_defaults_to_epsilon(self):
        p = UniformPerturbation(epsilon=0.2)
        assert p.cat_prob == pytest.approx(0.2)

    def test_no_cat_groups_backward_compat(self):
        p = UniformPerturbation(epsilon=0.1, continuous_indices=[0])
        x = torch.tensor([0.5, 1.0, 0.0, 0.0])
        x_prime = p(x)
        assert x_prime[1] == 1.0
        assert x_prime[2] == 0.0
        assert x_prime[3] == 0.0
