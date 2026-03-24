import torch
import torch.nn.functional as F
from src.utils.constants import GeometricDistanceType


class GeometricInstability:
    def __init__(self, cfs: torch.Tensor, cfs_prime: torch.Tensor):
        self.cfs = cfs
        self.cfs_prime = cfs_prime

        if self.cfs.shape != self.cfs_prime.shape:
            raise ValueError(
                f"`cfs` and `cfs_prime` must have the same shape, got "
                f"{self.cfs.shape} and {self.cfs_prime.shape}."
            )

        if self.cfs.ndim not in {1, 2}:
            raise ValueError(
                f"Expected `cfs` to be 1D or 2D, got ndim={self.cfs.ndim}."
            )

    def __call__(self, metric: GeometricDistanceType, inv_cov: torch.Tensor=None,
                 cosine_feature_dim: int=-1, reduction: str="none", eps: float=1e-8):
        if metric == GeometricDistanceType.L_1:
            return self._l_1(reduction=reduction)
        elif metric == GeometricDistanceType.L_2:
            return self._l_2(reduction=reduction)
        elif metric == GeometricDistanceType.L_INF:
            return self._l_inf(reduction=reduction)
        elif metric == GeometricDistanceType.COSINE:
            return self._cosine(dim=cosine_feature_dim, eps=eps, reduction=reduction)
        elif metric == GeometricDistanceType.MAHALANOBIS:
            if inv_cov is None:
                raise ValueError("`inv_cov` parameter must be provided for mahalanobis distance.")
            return self._mahalanobis(inv_cov=inv_cov, reduction=reduction)
        else:
            supported = ", ".join(m.value for m in GeometricDistanceType)
            raise ValueError(f"Unsupported metric {metric}. Supported metrics: {supported}")

    def _l_1(self, reduction: str="none") -> torch.Tensor:
        dist = torch.abs(self.cfs - self.cfs_prime)
        if self.cfs.ndim == 1:
            out = dist.sum()
        else:
            out = dist.sum(dim=1)

        out = self._reduce(out, reduction=reduction)
        return out

    def _l_2(self, reduction: str="none") -> torch.Tensor:
        diff = self.cfs - self.cfs_prime
        if self.cfs.ndim == 1:
            out = torch.linalg.norm(diff, ord=2)
        else:
            out = torch.linalg.norm(diff, ord=2, dim=1)

        out = self._reduce(out, reduction=reduction)
        return out
    
    def _l_inf(self, reduction: str="none") -> torch.Tensor:
        diff = torch.abs(self.cfs - self.cfs_prime)
        if self.cfs.ndim == 1:
            out = diff.max()
        else:
            out = diff.max(dim=1).values

        out = self._reduce(out, reduction=reduction)
        return out

    def _cosine(self, dim: int=-1, eps: float=1e-8, reduction: str="none") -> torch.Tensor:
        sim = F.cosine_similarity(self.cfs, self.cfs_prime, dim=dim, eps=eps)
        out = 1.0 - sim

        out = self._reduce(out, reduction=reduction)
        return out

    def _mahalanobis(self, inv_cov: torch.Tensor, reduction: str="none") -> torch.Tensor:
        d = self.cfs.shape[-1]
        if inv_cov.shape != (d, d):
            raise ValueError(
                f"`inv_cov` must have shape ({d}, {d}), got {inv_cov.shape}."
            )
        inv_cov = inv_cov.to(device=self.cfs.device, dtype=self.cfs.dtype)
        diff = self.cfs - self.cfs_prime
        if diff.ndim == 1:
            quad = diff @ inv_cov @ diff
            quad = torch.clamp(quad, min=0.0)
            out = torch.sqrt(quad)
        else:
            quad = torch.einsum("bi,ij,bj->b", diff, inv_cov, diff)
            quad = torch.clamp(quad, min=0.0)
            out = torch.sqrt(quad)

        out = self._reduce(out, reduction=reduction)
        return out

    def _reduce(self, out: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "none":
            return out
        elif reduction == "mean":
            return out.mean()
        elif reduction == "sum":
            return out.sum()
        raise ValueError(f"Unsupported reduction: {reduction}")
