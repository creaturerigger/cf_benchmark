from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePerturbation(ABC):
    """Abstract base for input perturbation strategies.

    A perturbation takes a query tensor x and returns a perturbed
    version x̃ of the same shape.
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a perturbed copy of *x*.

        Args:
            x: (D,) or (1, D) query tensor.

        Returns:
            Perturbed tensor with the same shape as *x*.
        """

    def generate(self, x: torch.Tensor, m: int) -> list[torch.Tensor]:
        """Generate *m* independent perturbations of *x*.

        Args:
            x: (D,) or (1, D) query tensor.
            m: number of perturbed copies to produce.

        Returns:
            List of *m* perturbed tensors, each with the same shape as *x*.
        """
        return [self(x) for _ in range(m)]

    @staticmethod
    def resample_categorical_groups(
        x: torch.Tensor,
        groups: list[list[int]],
        prob: float,
    ) -> torch.Tensor:
        """Randomly re-sample one-hot groups with probability *prob*.

        For each one-hot group, with probability ``prob`` a uniformly
        random category is activated (the rest are zeroed).  With
        probability ``1 - prob`` the group is left unchanged (Slack et al., 2021).

        Works for both (D,) and (B, D) shaped tensors.
        """
        for group_indices in groups:
            k = len(group_indices)
            if k < 2:
                continue  # nothing to perturb for a single-category group
            if torch.rand(1).item() < prob:
                new_cat = torch.randint(0, k, (1,)).item()
                x[..., group_indices] = 0.0
                x[..., group_indices[new_cat]] = 1.0
        return x
