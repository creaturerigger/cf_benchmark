from __future__ import annotations

from dataclasses import dataclass

import torch
from src.utils.constants import GeometricDistanceType


@dataclass
class MatchResult:
    """Holds index-aligned matched pairs within one query_uuid group."""
    original_indices: torch.Tensor   # (M,) indices into original pool
    prime_indices: torch.Tensor      # (M,) indices into prime pool
    distances: torch.Tensor          # (M,) distance of each matched pair


class NearestCFMatcher:
    """Intra-query matcher: pairs each original CF with its closest prime CF.

    Operates within a single (query_uuid, perturbation_id) group.
    """

    def __init__(
        self,
        metric: GeometricDistanceType = GeometricDistanceType.L_2,
        allow_many_to_one: bool = True,
    ) -> None:
        self.metric = metric
        self.allow_many_to_one = allow_many_to_one

    def match(self, pool: torch.Tensor,
              pool_prime: torch.Tensor) -> MatchResult:
        """Match each CF in *pool* to the nearest CF in *pool_prime*.

        Args:
            pool:       (N, D) tensor — original CF pool for one query_uuid.
            pool_prime: (M, D) tensor — prime CF pool for same query_uuid
                        and one perturbation_id.

        Returns:
            MatchResult with one entry per row in *pool*.
        """
        dist_matrix = self._pairwise_distances(pool, pool_prime)  # (N, M)

        if self.allow_many_to_one:
            return self._greedy_match(dist_matrix)
        return self._unique_match(dist_matrix)

    # ------------------------------------------------------------------
    # Matching strategies
    # ------------------------------------------------------------------

    def _greedy_match(self, dist_matrix: torch.Tensor) -> MatchResult:
        """Each original CF picks its nearest prime — primes may repeat."""
        distances, prime_indices = dist_matrix.min(dim=1)
        original_indices = torch.arange(dist_matrix.shape[0])
        return MatchResult(original_indices=original_indices,
                           prime_indices=prime_indices,
                           distances=distances)

    def _unique_match(self, dist_matrix: torch.Tensor) -> MatchResult:
        """1-to-1 matching: each prime is used at most once.

        Greedy row-wise assignment in order of increasing distance.
        If N > M some originals will be unmatched and excluded.
        """
        n, m = dist_matrix.shape
        flat = dist_matrix.flatten()
        sorted_indices = flat.argsort()

        used_originals = set()
        used_primes = set()
        orig_list: list[int] = []
        prime_list: list[int] = []
        dist_list: list[float] = []

        for idx in sorted_indices:
            i = int(idx // m)
            j = int(idx % m)
            if i in used_originals or j in used_primes:
                continue
            used_originals.add(i)
            used_primes.add(j)
            orig_list.append(i)
            prime_list.append(j)
            dist_list.append(float(dist_matrix[i, j]))
            if len(orig_list) == min(n, m):
                break

        return MatchResult(
            original_indices=torch.tensor(orig_list, dtype=torch.long),
            prime_indices=torch.tensor(prime_list, dtype=torch.long),
            distances=torch.tensor(dist_list),
        )

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _pairwise_distances(self, a: torch.Tensor,
                            b: torch.Tensor) -> torch.Tensor:
        """Compute (N, M) pairwise distance matrix between rows of a and b."""
        if self.metric == GeometricDistanceType.L_1:
            return torch.cdist(a.float(), b.float(), p=1.0)
        elif self.metric == GeometricDistanceType.L_2:
            return torch.cdist(a.float(), b.float(), p=2.0)
        elif self.metric == GeometricDistanceType.L_INF:
            return torch.cdist(a.float(), b.float(), p=float("inf"))
        elif self.metric == GeometricDistanceType.COSINE:
            a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
            b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
            return 1.0 - a_norm @ b_norm.T
        else:
            raise ValueError(f"Unsupported matching metric: {self.metric}")
