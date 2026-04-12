from typing import Any, List
import torch
from src.utils.constants import InterventionDistanceType


class InterventionStability:
    def __init__(self, cfs: torch.Tensor, cfs_prime: torch.Tensor,
                 x: torch.Tensor,
                 x_prime: torch.Tensor,
                 encoded_cont_feature_indices: list[int],
                 encoded_cat_feature_indices: List[List[int]]) -> None:
        self.x = x
        self.x_prime = x_prime
        self.cfs = cfs
        self.cfs_prime = cfs_prime
        self.encoded_continuous_feature_indices = encoded_cont_feature_indices
        self.encoded_categorical_feature_indices = encoded_cat_feature_indices
        self.cfs_changed = self._per_feature_change_vector(self.x,
                                                           self.cfs)
        self.cfs_prime_changed = \
            self._per_feature_change_vector(self.x_prime, self.cfs_prime)

    def __call__(self, metric: InterventionDistanceType) -> Any:
        if metric == InterventionDistanceType.JACCARD_INDEX:
            return self._jaccard_index()
        elif metric == InterventionDistanceType.DICE_SORENSEN_COEFFICIENT:
            return self._dice_sorensen_coefficient()

    def _jaccard_index(self) -> float:
        both_changed = (self.cfs_changed & self.cfs_prime_changed).sum().item()
        either_changed = (self.cfs_changed | self.cfs_prime_changed).sum() \
            .item()
        if either_changed == 0:
            return 0.0
        return 1.0 - (both_changed / (either_changed + 1e-8))

    def _dice_sorensen_coefficient(self) -> float:
        both_changed = (self.cfs_changed & self.cfs_prime_changed).sum().item()
        total_changed = self.cfs_changed.sum().item() + \
            self.cfs_prime_changed.sum().item()
        if total_changed == 0:
            return 0.0
        return 1.0 - ((2.0 * both_changed) / (total_changed + 1e-8))

    def _per_feature_change_vector(self, source: torch.Tensor,
                                   target: torch.Tensor) -> \
            torch.Tensor:

        cfs_cont_bins = self._bin_continuous_features(
            source, self.encoded_continuous_feature_indices)
        x_cont_bins = self._bin_continuous_features(
            target, self.encoded_continuous_feature_indices)
        cont_changed = cfs_cont_bins != x_cont_bins

        cat_changed_list = []
        for group in self.encoded_categorical_feature_indices:
            source_argmax = source[:, group].argmax(dim=1)
            target_argmax = target[:, group].argmax(dim=1)
            cat_changed_list.append(source_argmax != target_argmax)

        if cat_changed_list:
            cat_changed = torch.stack(cat_changed_list, dim=1)
            changed = torch.cat([cont_changed, cat_changed], dim=1)
        else:
            changed = cont_changed

        return changed

    def _bin_continuous_features(self, cfs: torch.Tensor,
                                 encoded_continuous_feature_indices: list[int],
                                 num_bins: int = 10) -> torch.Tensor:
        edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
        bins = []
        for cont_idx in encoded_continuous_feature_indices:
            col_vals = cfs[:, cont_idx].contiguous()
            binned = torch.bucketize(col_vals, edges, right=False) - 1
            binned = torch.clamp(binned, 0, num_bins - 1)
            bins.append(binned)
        return torch.stack(bins, dim=1)
