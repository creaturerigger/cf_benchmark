"""Multi-Objective Counterfactual (MOC) method — pymoo NSGA-II.

Re-implementation of Dandl et al. (2020) *Multi-Objective Counterfactual
Explanations* using pymoo's NSGA-II instead of the original R *ecr* package.

Four objectives (all minimised):
    0. dist_target  — |P(desired class | CF) − desired_class|
    1. dist_x       — Gower distance to the query instance
    2. nr_changed   — number of feature changes
    3. dist_train   — mean Gower distance to *k* nearest training points
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from sklearn.preprocessing import LabelEncoder

from .base_cf_method import BaseCounterfactualGenerationMethod
from .registry import register_method

logger = logging.getLogger(__name__)

_EPS = np.sqrt(np.finfo(float).eps)


# ────────────────────────────────────────────────────────────
# Result adapter
# ────────────────────────────────────────────────────────────
@dataclass
class MOCResult:
    """Adapter so pool builder can call ``result.to_dataframe()``."""

    cfs: pd.DataFrame | None
    feature_names: list[str]

    def to_dataframe(self) -> pd.DataFrame | None:
        return self.cfs


# ────────────────────────────────────────────────────────────
# Custom sampling: half the population starts near x.interest
# ────────────────────────────────────────────────────────────
class _NearQuerySampling(Sampling):
    """Seed half the initial population with random perturbations of
    *x.interest* (mimics the R ``traindata``/``random`` strategies)."""

    def __init__(self, x_interest: np.ndarray):
        super().__init__()
        self.x_interest = x_interest

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.xl, problem.xu
        X = np.random.uniform(xl, xu, size=(n_samples, problem.n_var))
        # Half of the population starts from x.interest with random changes
        n_near = n_samples // 2
        for i in range(n_near):
            X[i] = self.x_interest.copy()
            n_change = np.random.randint(1, problem.n_var + 1)
            idx = np.random.choice(problem.n_var, n_change, replace=False)
            for j in idx:
                X[i, j] = np.random.uniform(xl[j], xu[j])
        return X


# ────────────────────────────────────────────────────────────
# pymoo Problem — four MOC objectives
# ────────────────────────────────────────────────────────────
class _MOCProblem(Problem):
    """Four-objective optimisation problem for NSGA-II."""

    def __init__(
        self,
        x_interest: np.ndarray,
        predict_proba_fn,
        train_encoded: np.ndarray,
        n_feat: int,
        continuous_idx: list[int],
        categorical_idx: list[int],
        lower: np.ndarray,
        upper: np.ndarray,
        desired_class: int,
        k: int = 1,
    ):
        self.x_interest = x_interest
        self.predict_proba_fn = predict_proba_fn
        self.train_encoded = train_encoded
        self.continuous_idx = continuous_idx
        self.categorical_idx = categorical_idx
        self._cat_idx_set = set(categorical_idx)
        self._ranges = (upper - lower).copy()
        self._ranges[self._ranges == 0] = 1.0
        self.desired_class = desired_class
        self.k = k
        self.n_feat = n_feat
        super().__init__(n_var=n_feat, n_obj=4, xl=lower, xu=upper)

    # -- helpers --------------------------------------------------
    def _snap_categoricals(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx in self.categorical_idx:
            X[..., idx] = np.clip(
                np.round(X[..., idx]), self.xl[idx], self.xu[idx],
            )
        return X

    def _gower_distances(
        self, x: np.ndarray, Y: np.ndarray,
    ) -> np.ndarray:
        """Gower distance from *x* (1-D) to every row of *Y* (2-D)."""
        d = np.zeros(len(Y))
        for j in range(self.n_feat):
            if j in self._cat_idx_set:
                d += (np.round(x[j]) != np.round(Y[:, j])).astype(float)
            else:
                d += np.abs(x[j] - Y[:, j]) / self._ranges[j]
        return d / self.n_feat

    # -- evaluation -----------------------------------------------
    def _evaluate(self, X, out, *args, **kwargs):
        X_snap = self._snap_categoricals(X)
        n = len(X_snap)
        F = np.zeros((n, 4))
        x_ref = self.x_interest

        # Obj 0 — dist_target: |P(desired) − desired_class|
        probs = self.predict_proba_fn(X_snap)
        F[:, 0] = np.abs(probs - float(self.desired_class))

        # Obj 1 & 2 — Gower distance + nr_changed (shared loop)
        for j in range(self.n_feat):
            if j in self._cat_idx_set:
                changed = (
                    np.round(X_snap[:, j]) != np.round(x_ref[j])
                ).astype(float)
                F[:, 1] += changed          # Gower: 0/1
                F[:, 2] += changed          # nr changed
            else:
                abs_diff = np.abs(X_snap[:, j] - x_ref[j])
                F[:, 1] += abs_diff / self._ranges[j]   # Gower: normalised
                F[:, 2] += (abs_diff >= _EPS).astype(float)
        F[:, 1] /= self.n_feat

        # Obj 3 — dist_train: mean Gower distance to k nearest neighbours
        for i in range(n):
            dists = self._gower_distances(X_snap[i], self.train_encoded)
            if self.k == 1:
                F[i, 3] = np.min(dists)
            else:
                F[i, 3] = np.mean(np.partition(dists, self.k)[: self.k])

        out["F"] = F


# ────────────────────────────────────────────────────────────
# Public wrapper
# ────────────────────────────────────────────────────────────
@register_method(name="moc")
class MOCMethod(BaseCounterfactualGenerationMethod):
    """MOC wrapper around pymoo NSGA-II."""

    def __init__(
        self,
        cfg: dict,
        model,
        dataframe: pd.DataFrame,
        target_column: str,
        continuous_features: list[str],
    ):
        super().__init__(cfg)

        self.target_column = target_column
        feature_cols = [c for c in dataframe.columns if c != target_column]
        self.feature_names = feature_cols
        self._continuous = continuous_features
        self._categorical = [
            c for c in feature_cols if c not in continuous_features
        ]

        self._continuous_idx = [feature_cols.index(c) for c in self._continuous]
        self._categorical_idx = [feature_cols.index(c) for c in self._categorical]

        # Label-encode categoricals
        self._label_encoders: dict[str, LabelEncoder] = {}
        train_df = dataframe[feature_cols].copy()
        for col in self._categorical:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            self._label_encoders[col] = le

        self._train_encoded = train_df.values.astype(np.float64)
        self._lower = self._train_encoded.min(axis=0)
        self._upper = self._train_encoded.max(axis=0)

        self._model = model
        self._build_predict_proba_fn(model, dataframe, target_column, feature_cols)
        self._moc_cfg = cfg.get("moc", {})

    # ── predict_fn bridge ────────────────────────────────────
    def _build_predict_proba_fn(
        self, model, dataframe, target_column, feature_cols,
    ) -> None:
        """Label-encoded array → P(class = 1) via OHE + scaler + model."""
        from src.data.preprocessing.py_dataset import PYTDataset

        tmp_ds = PYTDataset(
            dataframe=dataframe,
            target_column=target_column,
            train=True,
        )
        self._scaler = tmp_ds.scaler
        self._ohe = tmp_ds.encoder

        scaler_cols = list(self._scaler.feature_names_in_)
        ohe_cols = list(self._ohe.feature_names_in_) if self._ohe is not None else []
        categorical = self._categorical

        def predict_proba_fn(X: np.ndarray) -> np.ndarray:
            """(n, d) label-encoded → (n,) P(class = 1)."""
            X = np.atleast_2d(X)
            df = pd.DataFrame(X, columns=feature_cols)
            for col in categorical:
                le = self._label_encoders[col]
                df[col] = le.inverse_transform(df[col].round().astype(int))

            parts = []
            if scaler_cols:
                parts.append(self._scaler.transform(df[scaler_cols]))
            if ohe_cols and self._ohe is not None:
                parts.append(self._ohe.transform(df[ohe_cols]))
            encoded = np.concatenate(parts, axis=1).astype(np.float32)

            t = torch.tensor(encoded, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                logits = model(t)
            return logits.squeeze(-1).numpy()

        self._predict_proba_fn = predict_proba_fn

    # ── generate ─────────────────────────────────────────────
    def generate(
        self,
        query_instance: pd.DataFrame,
        num_cfs: int,
        **kwargs,
    ) -> MOCResult:
        feat_cols = self.feature_names
        q = query_instance[feat_cols].copy()
        for col in self._categorical:
            le = self._label_encoders[col]
            q[col] = le.transform(q[col].astype(str))
        x_interest = q.values.astype(np.float64).ravel()

        # Determine desired class (opposite of current prediction)
        current_prob = self._predict_proba_fn(x_interest.reshape(1, -1))[0]
        current_class = int(current_prob >= 0.5)
        desired_class = 1 - current_class

        moc = self._moc_cfg
        pop_size = moc.get("pop_size", 20)
        n_gen = moc.get("n_gen", 175)
        k = moc.get("k", 1)

        problem = _MOCProblem(
            x_interest=x_interest,
            predict_proba_fn=self._predict_proba_fn,
            train_encoded=self._train_encoded,
            n_feat=len(feat_cols),
            continuous_idx=self._continuous_idx,
            categorical_idx=self._categorical_idx,
            lower=self._lower.copy(),
            upper=self._upper.copy(),
            desired_class=desired_class,
            k=k,
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=_NearQuerySampling(x_interest),
            crossover=SBX(
                prob=moc.get("p_cross", 0.57),
                eta=moc.get("eta_cross", 15),
            ),
            mutation=PM(
                prob=moc.get("p_mut", 0.79),
                eta=moc.get("eta_mut", 20),
            ),
        )

        try:
            res = pymoo_minimize(
                problem,
                algorithm,
                termination=("n_gen", n_gen),
                seed=moc.get("seed", None),
                verbose=False,
            )
        except Exception:
            logger.warning("MOC NSGA-II optimisation failed", exc_info=True)
            return MOCResult(cfs=None, feature_names=self.feature_names)

        if res.F is None or res.X is None:
            return MOCResult(cfs=None, feature_names=self.feature_names)

        # Select valid CFs from the Pareto front
        X_pareto = problem._snap_categoricals(res.X)
        probs = self._predict_proba_fn(X_pareto)
        if desired_class == 1:
            valid_mask = probs >= 0.5
        else:
            valid_mask = probs < 0.5

        if valid_mask.any():
            X_valid = X_pareto[valid_mask]
            F_valid = res.F[valid_mask]
            order = np.lexsort((F_valid[:, 1], F_valid[:, 0]))
            X_selected = X_valid[order]
        else:
            # Fall back: all Pareto-optimal, sorted by dist_target
            X_selected = X_pareto[np.argsort(res.F[:, 0])]

        X_selected = X_selected[: num_cfs]

        # Decode back to original feature space
        cfs_df = pd.DataFrame(X_selected, columns=self.feature_names)
        for col in self._categorical:
            le = self._label_encoders[col]
            cfs_df[col] = le.inverse_transform(cfs_df[col].round().astype(int))
        for col in self._continuous:
            cfs_df[col] = cfs_df[col].astype(float)

        return MOCResult(cfs=cfs_df, feature_names=self.feature_names)
