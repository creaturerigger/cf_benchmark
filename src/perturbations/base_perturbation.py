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
