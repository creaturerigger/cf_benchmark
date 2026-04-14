from __future__ import annotations

import torch
from src.perturbations.base_perturbation import BasePerturbation


class GaussianPerturbation(BasePerturbation):
    """Additive Gaussian noise on continuous features, random re-sampling
    on categorical (one-hot) features.

    Continuous: x̃_cont = x_cont + δ,  δ ~ N(0, σ²I), clamped to [0, 1].
    Categorical: for each one-hot group, with probability ``cat_prob``
        the active category is replaced by a uniformly random category
        from the same group.  Default ``cat_prob = sigma`` so a single
        σ knob controls both perturbation strengths (Slack et al., 2021).

    Args:
        sigma: standard deviation of the Gaussian noise.
        continuous_indices: indices of continuous columns in the
            encoded tensor.  If ``None`` all columns are treated
            as continuous (no categorical perturbation).
        categorical_groups: list of index-lists, one per categorical
            feature.  Each inner list contains the tensor column
            indices for that feature's one-hot encoding.
            If ``None`` categorical columns are left unchanged.
        cat_prob: probability of re-sampling each categorical group.
            Defaults to ``None`` which uses ``sigma`` (clamped to [0, 1]).
        clamp_min: lower clamp bound (default 0.0).
        clamp_max: upper clamp bound (default 1.0).
    """

    def __init__(
        self,
        sigma: float,
        continuous_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        cat_prob: float | None = None,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.continuous_indices = continuous_indices
        self.categorical_groups = categorical_groups
        self.cat_prob = cat_prob if cat_prob is not None else min(max(sigma, 0.0), 1.0)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_prime = x.clone()

        # ── Continuous perturbation ──────────────────────────
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

        # ── Categorical perturbation (one-hot re-sampling) ───
        if self.categorical_groups and self.cat_prob > 0:
            x_prime = self.resample_categorical_groups(
                x_prime, self.categorical_groups, self.cat_prob,
            )

        return x_prime
