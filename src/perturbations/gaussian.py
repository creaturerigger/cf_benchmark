from __future__ import annotations

import torch

from src.perturbations.base_perturbation import BasePerturbation


class GaussianPerturbation(BasePerturbation):
    """Additive Gaussian noise: x̃ = x + δ, δ ~ N(0, σ²I).

    Continuous features are perturbed and clamped to [0, 1].
    Categorical features (one-hot groups) are left unchanged.

    Args:
        sigma: standard deviation of the Gaussian noise.
        continuous_indices: indices of continuous columns in the
            encoded tensor.  If ``None`` all columns are treated
            as continuous.
        clamp_min: lower clamp bound (default 0.0).
        clamp_max: upper clamp bound (default 1.0).
    """

    def __init__(
        self,
        sigma: float,
        continuous_indices: list[int] | None = None,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.continuous_indices = continuous_indices
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_prime = x.clone()

        if self.continuous_indices is not None:
            idx = self.continuous_indices
            noise = torch.randn(
                *x_prime[..., idx].shape,
                dtype=x.dtype,
                device=x.device,
            ) * self.sigma
            x_prime[..., idx] = (x_prime[..., idx] + noise).clamp(
                self.clamp_min, self.clamp_max,
            )
        else:
            noise = torch.randn_like(x_prime) * self.sigma
            x_prime = (x_prime + noise).clamp(self.clamp_min, self.clamp_max)

        return x_prime
