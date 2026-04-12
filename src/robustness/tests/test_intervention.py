import torch
import pytest

from src.robustness.intervention import InterventionStability
from src.utils.constants import InterventionDistanceType


def _make_stability(
    x: torch.Tensor,
    x_prime: torch.Tensor,
    cfs: torch.Tensor,
    cfs_prime: torch.Tensor,
    cont_indices: list[int],
    cat_indices: list[list[int]],
) -> InterventionStability:
    return InterventionStability(
        cfs=cfs,
        cfs_prime=cfs_prime,
        x=x,
        x_prime=x_prime,
        encoded_cont_feature_indices=cont_indices,
        encoded_cat_feature_indices=cat_indices,
    )


class TestJaccardIndex:
    def test_identical_changes_returns_zero(self):
        """Same recourse strategy → Jaccard dissimilarity = 0."""
        # 2 CFs, 3 continuous features, no categoricals
        x = torch.tensor([[0.1, 0.2, 0.3]])
        x_prime = torch.tensor([[0.1, 0.2, 0.3]])
        cfs = torch.tensor([[0.9, 0.2, 0.3]])       # feature 0 changed
        cfs_prime = torch.tensor([[0.9, 0.2, 0.3]])  # same change
        st = _make_stability(x, x_prime, cfs, cfs_prime, [0, 1, 2], [])
        result = st(InterventionDistanceType.JACCARD_INDEX)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_completely_different_changes_returns_one(self):
        """Disjoint change sets → Jaccard dissimilarity = 1."""
        x = torch.tensor([[0.5, 0.5, 0.5]])
        x_prime = torch.tensor([[0.5, 0.5, 0.5]])
        # original changes feature 0 only (big jump crosses bin boundary)
        cfs = torch.tensor([[0.05, 0.5, 0.5]])
        # prime changes feature 2 only
        cfs_prime = torch.tensor([[0.5, 0.5, 0.05]])
        st = _make_stability(x, x_prime, cfs, cfs_prime, [0, 1, 2], [])
        result = st(InterventionDistanceType.JACCARD_INDEX)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_no_changes_returns_zero(self):
        """If neither recourse changes anything, dissimilarity = 0."""
        x = torch.tensor([[0.5, 0.5]])
        x_prime = torch.tensor([[0.5, 0.5]])
        cfs = torch.tensor([[0.5, 0.5]])
        cfs_prime = torch.tensor([[0.5, 0.5]])
        st = _make_stability(x, x_prime, cfs, cfs_prime, [0, 1], [])
        result = st(InterventionDistanceType.JACCARD_INDEX)
        assert result == 0.0


class TestDiceSorensenCoefficient:
    def test_identical_changes_returns_zero(self):
        x = torch.tensor([[0.1, 0.2, 0.3]])
        x_prime = torch.tensor([[0.1, 0.2, 0.3]])
        cfs = torch.tensor([[0.9, 0.2, 0.3]])
        cfs_prime = torch.tensor([[0.9, 0.2, 0.3]])
        st = _make_stability(x, x_prime, cfs, cfs_prime, [0, 1, 2], [])
        result = st(InterventionDistanceType.DICE_SORENSEN_COEFFICIENT)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_completely_different_changes_returns_one(self):
        x = torch.tensor([[0.5, 0.5, 0.5]])
        x_prime = torch.tensor([[0.5, 0.5, 0.5]])
        cfs = torch.tensor([[0.05, 0.5, 0.5]])
        cfs_prime = torch.tensor([[0.5, 0.5, 0.05]])
        st = _make_stability(x, x_prime, cfs, cfs_prime, [0, 1, 2], [])
        result = st(InterventionDistanceType.DICE_SORENSEN_COEFFICIENT)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_no_changes_returns_zero(self):
        x = torch.tensor([[0.5, 0.5]])
        x_prime = torch.tensor([[0.5, 0.5]])
        cfs = torch.tensor([[0.5, 0.5]])
        cfs_prime = torch.tensor([[0.5, 0.5]])
        st = _make_stability(x, x_prime, cfs, cfs_prime, [0, 1], [])
        result = st(InterventionDistanceType.DICE_SORENSEN_COEFFICIENT)
        assert result == 0.0


class TestBinContinuousFeatures:
    def test_same_value_same_bin(self):
        """Values in the same bin should not register as changed."""
        x = torch.tensor([[0.51]])
        cfs = torch.tensor([[0.52]])  # tiny shift, same bin
        st = _make_stability(x, x, cfs, cfs, [0], [])
        # Both should have same bin → no change detected
        assert st.cfs_changed.sum().item() == 0

    def test_different_bins_detected(self):
        """Values in different bins should register as changed."""
        x = torch.tensor([[0.05]])
        cfs = torch.tensor([[0.95]])  # large jump, different bin
        st = _make_stability(x, x, cfs, cfs, [0], [])
        assert st.cfs_changed.sum().item() > 0


class TestWithCategoricalFeatures:
    def test_same_categorical_change(self):
        """Same categorical feature changed the same way → no dissimilarity."""
        # 1 continuous + 1 categorical (one-hot with 3 classes: cols 1,2,3)
        x = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        x_prime = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        # Both change category from 0 to 1
        cfs = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
        cfs_prime = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
        st = _make_stability(
            x, x_prime, cfs, cfs_prime,
            cont_indices=[0],
            cat_indices=[[1, 2, 3]],
        )
        result = st(InterventionDistanceType.JACCARD_INDEX)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_different_categorical_change(self):
        """One changes category, other doesn't → nonzero dissimilarity."""
        x = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        x_prime = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        # Original changes category from 0 to 1
        cfs = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
        # Prime does NOT change category (stays 0)
        cfs_prime = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        st = _make_stability(
            x, x_prime, cfs, cfs_prime,
            cont_indices=[0],
            cat_indices=[[1, 2, 3]],
        )
        result = st(InterventionDistanceType.JACCARD_INDEX)
        assert result > 0.0


class TestOutputRange:
    def test_jaccard_bounded(self):
        x = torch.rand(1, 5)
        x_prime = x + torch.rand(1, 5) * 0.5
        cfs = torch.rand(1, 5)
        cfs_prime = torch.rand(1, 5)
        st = _make_stability(
            x, x_prime, cfs, cfs_prime,
            cont_indices=list(range(5)), cat_indices=[],
        )
        result = st(InterventionDistanceType.JACCARD_INDEX)
        assert 0.0 <= result <= 1.0 + 1e-6

    def test_dice_bounded(self):
        x = torch.rand(1, 5)
        x_prime = x + torch.rand(1, 5) * 0.5
        cfs = torch.rand(1, 5)
        cfs_prime = torch.rand(1, 5)
        st = _make_stability(
            x, x_prime, cfs, cfs_prime,
            cont_indices=list(range(5)), cat_indices=[],
        )
        result = st(InterventionDistanceType.DICE_SORENSEN_COEFFICIENT)
        assert 0.0 <= result <= 1.0 + 1e-6
