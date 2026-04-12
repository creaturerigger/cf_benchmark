import torch
import pytest

from src.robustness.matcher import NearestCFMatcher, MatchResult
from src.utils.constants import GeometricDistanceType


class TestGreedyMatch:
    def test_identical_pools_zero_distance(self):
        pool = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        matcher = NearestCFMatcher(metric=GeometricDistanceType.L_2)
        result = matcher.match(pool, pool.clone())
        assert torch.allclose(result.distances, torch.zeros(2), atol=1e-6)
        assert torch.equal(result.original_indices, torch.arange(2))
        assert torch.equal(result.prime_indices, torch.arange(2))

    def test_known_matching(self):
        pool = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        pool_prime = torch.tensor([[10.0, 10.0], [0.0, 0.0]])
        matcher = NearestCFMatcher(metric=GeometricDistanceType.L_2)
        result = matcher.match(pool, pool_prime)
        # row 0 should match prime row 1, row 1 should match prime row 0
        assert result.prime_indices[0].item() == 1
        assert result.prime_indices[1].item() == 0

    def test_many_to_one_allowed(self):
        pool = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
        pool_prime = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        matcher = NearestCFMatcher(
            metric=GeometricDistanceType.L_2, allow_many_to_one=True,
        )
        result = matcher.match(pool, pool_prime)
        # Both should match the nearest prime (index 0)
        assert result.prime_indices[0].item() == 0
        assert result.prime_indices[1].item() == 0

    def test_output_shapes(self):
        pool = torch.rand(5, 3)
        pool_prime = torch.rand(4, 3)
        matcher = NearestCFMatcher()
        result = matcher.match(pool, pool_prime)
        assert result.original_indices.shape == (5,)
        assert result.prime_indices.shape == (5,)
        assert result.distances.shape == (5,)


class TestUniqueMatch:
    def test_one_to_one(self):
        pool = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
        pool_prime = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        matcher = NearestCFMatcher(
            metric=GeometricDistanceType.L_2, allow_many_to_one=False,
        )
        result = matcher.match(pool, pool_prime)
        # No prime index should repeat
        assert len(set(result.prime_indices.tolist())) == len(
            result.prime_indices
        )

    def test_fewer_primes_limits_matches(self):
        pool = torch.rand(5, 3)
        pool_prime = torch.rand(2, 3)
        matcher = NearestCFMatcher(
            metric=GeometricDistanceType.L_2, allow_many_to_one=False,
        )
        result = matcher.match(pool, pool_prime)
        assert len(result.original_indices) == 2  # min(5, 2)

    def test_square_case_all_matched(self):
        pool = torch.rand(4, 3)
        pool_prime = torch.rand(4, 3)
        matcher = NearestCFMatcher(
            metric=GeometricDistanceType.L_2, allow_many_to_one=False,
        )
        result = matcher.match(pool, pool_prime)
        assert len(result.original_indices) == 4
        assert len(set(result.prime_indices.tolist())) == 4


class TestMetrics:
    @pytest.fixture()
    def pools(self):
        pool = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pool_prime = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        return pool, pool_prime

    def test_l1(self, pools):
        matcher = NearestCFMatcher(metric=GeometricDistanceType.L_1)
        result = matcher.match(*pools)
        assert torch.allclose(result.distances, torch.zeros(2), atol=1e-6)

    def test_l2(self, pools):
        matcher = NearestCFMatcher(metric=GeometricDistanceType.L_2)
        result = matcher.match(*pools)
        assert torch.allclose(result.distances, torch.zeros(2), atol=1e-6)

    def test_l_inf(self, pools):
        matcher = NearestCFMatcher(metric=GeometricDistanceType.L_INF)
        result = matcher.match(*pools)
        assert torch.allclose(result.distances, torch.zeros(2), atol=1e-6)

    def test_cosine(self, pools):
        matcher = NearestCFMatcher(metric=GeometricDistanceType.COSINE)
        result = matcher.match(*pools)
        assert torch.allclose(result.distances, torch.zeros(2), atol=1e-5)

    def test_unsupported_metric_raises(self):
        matcher = NearestCFMatcher(metric=GeometricDistanceType.MAHALANOBIS)
        with pytest.raises(ValueError, match="Unsupported"):
            matcher.match(torch.rand(2, 3), torch.rand(2, 3))


class TestMatchResult:
    def test_dataclass_fields(self):
        mr = MatchResult(
            original_indices=torch.tensor([0, 1]),
            prime_indices=torch.tensor([1, 0]),
            distances=torch.tensor([0.5, 0.3]),
        )
        assert mr.original_indices.shape == (2,)
        assert mr.prime_indices.shape == (2,)
        assert mr.distances.shape == (2,)
