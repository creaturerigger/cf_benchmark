from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from src.robustness.geometric import GeometricInstability
from src.robustness.intervention import InterventionStability
from src.robustness.matcher import NearestCFMatcher
from src.robustness.score import (
    CandidateObjectives,
    pareto_front,
)
from src.utils.constants import GeometricDistanceType, InterventionDistanceType


@dataclass
class QueryResult:
    query_uuid: str
    sigma: float
    candidates: List[CandidateObjectives]
    pareto_front: List[CandidateObjectives]


class RobustnessExperiment:
    def __init__(
        self,
        matcher: NearestCFMatcher,
        geometric_metric: GeometricDistanceType = GeometricDistanceType.L_1,
        intervention_metric: InterventionDistanceType = (
            InterventionDistanceType.JACCARD_INDEX
        ),
        encoded_cont_feature_indices: Optional[list[int]] = None,
        encoded_cat_feature_indices: Optional[List[List[int]]] = None,
    ) -> None:
        self.matcher = matcher
        self.geometric_metric = geometric_metric
        self.intervention_metric = intervention_metric
        self.cont_indices = encoded_cont_feature_indices or []
        self.cat_indices = encoded_cat_feature_indices or []

    def evaluate_query(
        self,
        query_uuid: str,
        sigma: float,
        x: torch.Tensor,
        pool: torch.Tensor,
        perturbed_queries: List[torch.Tensor],
        perturbed_pools: List[torch.Tensor],
    ) -> QueryResult:
        """Evaluate all candidates in *pool* for one query UUID and sigma.

        Args:
            query_uuid: identifier for the original query.
            sigma: perturbation magnitude used.
            x: (1, D) or (D,) original query tensor.
            pool: (N, D) original CF pool tensor.
            perturbed_queries: list of M perturbed query tensors.
            perturbed_pools: list of M corresponding prime CF pool tensors.

        Returns:
            QueryResult with per-candidate objectives and the Pareto front.
        """
        n = pool.shape[0]
        x_2d = x.unsqueeze(0) if x.ndim == 1 else x

        # Proximity: distance from each candidate to the original query
        proximity_scores = GeometricInstability(
            cfs=pool, cfs_prime=x_2d.expand(n, -1),
        )(self.geometric_metric, reduction="none")

        # Robustness: average over perturbations
        geo_accum = torch.zeros(n)
        int_accum = torch.zeros(n)
        m_count = 0

        for x_prime, pool_prime in zip(perturbed_queries, perturbed_pools):
            if pool_prime.shape[0] == 0:
                continue

            match = self.matcher.match(pool, pool_prime)
            matched_pool = pool[match.original_indices]
            matched_prime = pool_prime[match.prime_indices]

            # Geometric instability per matched pair
            geo = GeometricInstability(
                cfs=matched_pool, cfs_prime=matched_prime,
            )(self.geometric_metric, reduction="none")

            # Intervention instability per matched pair
            x_prime_2d = x_prime.unsqueeze(0) if x_prime.ndim == 1 else x_prime

            int_scores = torch.zeros(len(match.original_indices))
            for k in range(len(match.original_indices)):
                c = matched_pool[k].unsqueeze(0)
                c_prime = matched_prime[k].unsqueeze(0)
                x_row = x_2d
                xp_row = x_prime_2d

                int_scores[k] = InterventionStability(
                    cfs=c, cfs_prime=c_prime,
                    x=x_row, x_prime=xp_row,
                    encoded_cont_feature_indices=self.cont_indices,
                    encoded_cat_feature_indices=self.cat_indices,
                )(self.intervention_metric)

            # Scatter back to full pool indices
            for idx_pos, orig_idx in enumerate(match.original_indices):
                geo_accum[orig_idx] += geo[idx_pos]
                int_accum[orig_idx] += int_scores[idx_pos]

            m_count += 1

        if m_count > 0:
            geo_accum /= m_count
            int_accum /= m_count

        candidates = []
        for i in range(n):
            candidates.append(CandidateObjectives(
                query_uuid=query_uuid,
                cf_index=i,
                proximity=float(proximity_scores[i]),
                geometric_instability=float(geo_accum[i]),
                intervention_instability=float(int_accum[i]),
            ))

        front = pareto_front(candidates)

        return QueryResult(
            query_uuid=query_uuid,
            sigma=sigma,
            candidates=candidates,
            pareto_front=front,
        )

    def run(
        self,
        queries: dict,
        sigma: float,
    ) -> List[QueryResult]:
        """Run evaluation across all query UUIDs.

        Args:
            queries: mapping of query_uuid to a dict with keys:
                - "x": original query tensor
                - "pool": original CF pool tensor
                - "perturbed_queries": list of perturbed query tensors
                - "perturbed_pools": list of prime pool tensors
            sigma: perturbation magnitude.

        Returns:
            List of QueryResult, one per query UUID.
        """
        results = []
        for query_uuid, data in queries.items():
            result = self.evaluate_query(
                query_uuid=query_uuid,
                sigma=sigma,
                x=data["x"],
                pool=data["pool"],
                perturbed_queries=data["perturbed_queries"],
                perturbed_pools=data["perturbed_pools"],
            )
            results.append(result)
        return results

    def to_records(self, results: List[QueryResult]) -> list[dict]:
        """Flatten results into a list of dicts for downstream aggregation."""
        records = []
        for qr in results:
            pareto_indices = {c.cf_index for c in qr.pareto_front}
            for c in qr.candidates:
                records.append({
                    "query_uuid": c.query_uuid,
                    "cf_index": c.cf_index,
                    "sigma": qr.sigma,
                    "proximity": c.proximity,
                    "geometric_instability": c.geometric_instability,
                    "intervention_instability": c.intervention_instability,
                    "is_pareto_optimal": c.cf_index in pareto_indices,
                })
        return records
