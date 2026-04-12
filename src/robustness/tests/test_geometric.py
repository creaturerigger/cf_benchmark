import pytest
import torch

from src.robustness.geometric import GeometricInstability
from src.utils.constants import GeometricDistanceType


class TestInit:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            GeometricInstability(torch.rand(3, 4), torch.rand(2, 4))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="1D or 2D"):
            GeometricInstability(torch.rand(2, 3, 4), torch.rand(2, 3, 4))

    def test_valid_2d(self):
        gi = GeometricInstability(torch.rand(3, 4), torch.rand(3, 4))
        assert gi.cfs.shape == (3, 4)

    def test_valid_1d(self):
        gi = GeometricInstability(torch.rand(4), torch.rand(4))
        assert gi.cfs.shape == (4,)


class TestL1:
    def test_identical_is_zero(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gi = GeometricInstability(t, t.clone())
        result = gi(GeometricDistanceType.L_1)
        assert torch.allclose(result, torch.zeros(2))

    def test_known_values(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[1.0, 2.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_1)
        assert torch.allclose(result, torch.tensor([3.0]))

    def test_1d(self):
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([3.0, 4.0])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_1)
        assert torch.allclose(result, torch.tensor(7.0))

    def test_reduction_mean(self):
        a = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_1, reduction="mean")
        assert torch.allclose(result, torch.tensor(2.0))

    def test_reduction_sum(self):
        a = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_1, reduction="sum")
        assert torch.allclose(result, torch.tensor(4.0))


class TestL2:
    def test_identical_is_zero(self):
        t = torch.rand(5, 3)
        gi = GeometricInstability(t, t.clone())
        result = gi(GeometricDistanceType.L_2)
        assert torch.allclose(result, torch.zeros(5), atol=1e-7)

    def test_known_values(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[3.0, 4.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_2)
        assert torch.allclose(result, torch.tensor([5.0]))

    def test_1d(self):
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([3.0, 4.0])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_2)
        assert torch.allclose(result, torch.tensor(5.0))


class TestLInf:
    def test_known_values(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[3.0, 4.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_INF)
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_1d(self):
        a = torch.tensor([1.0, 5.0, 2.0])
        b = torch.tensor([1.0, 0.0, 2.0])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.L_INF)
        assert torch.allclose(result, torch.tensor(5.0))


class TestCosine:
    def test_parallel_vectors_zero_distance(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[2.0, 0.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.COSINE)
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_orthogonal_vectors(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.COSINE)
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-6)


class TestMahalanobis:
    def test_identity_cov_equals_l2(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[3.0, 4.0]])
        inv_cov = torch.eye(2)
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.MAHALANOBIS, inv_cov=inv_cov)
        assert torch.allclose(result, torch.tensor([5.0]), atol=1e-5)

    def test_missing_inv_cov_raises(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        gi = GeometricInstability(a, b)
        with pytest.raises(ValueError, match="inv_cov"):
            gi(GeometricDistanceType.MAHALANOBIS)

    def test_wrong_inv_cov_shape_raises(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        gi = GeometricInstability(a, b)
        with pytest.raises(ValueError, match="inv_cov"):
            gi(GeometricDistanceType.MAHALANOBIS, inv_cov=torch.eye(3))

    def test_1d(self):
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([3.0, 4.0])
        inv_cov = torch.eye(2)
        gi = GeometricInstability(a, b)
        result = gi(GeometricDistanceType.MAHALANOBIS, inv_cov=inv_cov)
        assert torch.allclose(result, torch.tensor(5.0), atol=1e-5)


class TestReduction:
    def test_invalid_reduction_raises(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        gi = GeometricInstability(a, b)
        with pytest.raises(ValueError, match="Unsupported reduction"):
            gi(GeometricDistanceType.L_1, reduction="max")
