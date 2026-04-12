from __future__ import annotations

import torch

from src.perturbations.base_perturbation import BasePerturbation


class UniformPerturbation(BasePerturbation):
    """Additive uniform noise: x̃ = x + δ, δ ~ U(-ε, ε).

    Continuous features are perturbed and clamped to [0, 1].
    Categorical features (one-hot groups) are left unchanged.

    Args:
        epsilon: half-width of the uniform interval.
        continuous_indices: indices of continuous columns in the
            encoded tensor.  If ``None`` all columns are treated
            as continuous.
        clamp_min: lower clamp bound (default 0.0).
        clamp_max: upper clamp bound (default 1.0).
    """

    def __init__(
        self,
        epsilon: float,
        continuous_indices: list[int] | None = None,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        self.epsilon = epsilon
        self.continuous_indices = continuous_indices
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_prime = x.clone()

        if self.continuous_indices is not None:
            idx = self.continuous_indices
            noise = (
                torch.rand(
                    *x_prime[..., idx].shape,
                    dtype=x.dtype,
                    device=x.device,
                ) * 2 - 1
            ) * self.epsilon
            x_prime[..., idx] = (x_prime[..., idx] + noise).clamp(
                self.clamp_min, self.clamp_max,
            )
        else:
            noise = (torch.rand_like(x_prime) * 2 - 1) * self.epsilon
            x_prime = (x_prime + noise).clamp(self.clamp_min, self.clamp_max)

        return x_prime
