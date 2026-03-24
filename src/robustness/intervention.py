from typing import Any
import torch
from src.utils.constants import InterventionDistanceType
from typing import List

class InterventionInstability:
    def __init__(self, cfs: torch.Tensor, cfs_prime: torch.Tensor) -> None:
        self.cfs = cfs
        self.cfs_prime = cfs_prime

    def __call__(self, metric: InterventionDistanceType) -> Any:
        if metric == InterventionDistanceType.JACCARD_INDEX:
            return self._jaccard_index()
        elif metric == InterventionDistanceType.DICE_SORENSEN_COEFFICIENT:
            return self._dice_sorensen_coefficient()

    def _jaccard_index(self):
        pass

    def _dice_sorensen_coefficient(self):
        pass

    def _preprocess_for_computation(self, cfs: torch.Tensor,
                                    encoded_categorical_feature_indices: List[List[int]]):

        cat_col_indices = [col for group in encoded_categorical_feature_indices for col in group]



    def _binarize_continuous_features(self, cfs: torch.Tensor,
                                      encoded_continuous_feature_indices: list[int],
                                      num_bins: int=10) -> torch.Tensor:
        edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
        all_one_hots = []
        for cont_idx in encoded_continuous_feature_indices:
            col_vals = cfs[:, cont_idx].contiguous()
            binned_indices = torch.bucketize(col_vals, edges, right=False) - 1
            binned_indices = torch.clamp(binned_indices, 0, num_bins - 1)

            one_hot_col = torch.nn.functional.one_hot(binned_indices,
                                                        num_classes=num_bins).float()
            all_one_hots.append(one_hot_col)
        return torch.cat(all_one_hots, dim=1)
