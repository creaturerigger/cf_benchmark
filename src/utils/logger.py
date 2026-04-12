from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


LOGS_DIR = Path("results/logs")


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    logs_dir: Optional[Path] = None,
) -> logging.Logger:
    """Return a logger that writes to console and optionally to a file.

    Args:
        name: logger name (typically __name__ or the experiment id).
        level: logging level.
        log_to_file: if True, also write to results/logs/<name>.log.
        logs_dir: override the default LOGS_DIR.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_to_file:
        dest = logs_dir or LOGS_DIR
        dest.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(dest / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class ExperimentLogger:
    """Structured logger for reproducibility-critical information.

    Writes a JSON-lines file alongside human-readable logs so that
    every decision can be traced back when reviewers ask.
    """

    def __init__(self, experiment_id: str, logs_dir: Optional[Path] = None) -> None:
        self.experiment_id = experiment_id
        self._logs_dir = logs_dir or LOGS_DIR
        self.logger = get_logger(experiment_id, logs_dir=self._logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self._logs_dir / f"{experiment_id}.jsonl"

    # ── low-level ────────────────────────────────────────────

    def _write_event(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "event": event,
            **data,
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── experiment lifecycle ─────────────────────────────────

    def log_config(self, cfg: dict[str, Any]) -> None:
        """Log the full merged config at experiment start."""
        self.logger.info("Config: %s", json.dumps(cfg, default=str))
        self._write_event("config", {"config": cfg})

    def log_seed(self, seed: int) -> None:
        self.logger.info("Random seed: %d", seed)
        self._write_event("seed", {"seed": seed})

    def log_dataset(
        self, name: str, n_rows: int, n_features: int,
        target: str, class_distribution: dict,
    ) -> None:
        self.logger.info(
            "Dataset %s: %d rows, %d features, target=%s, dist=%s",
            name, n_rows, n_features, target, class_distribution,
        )
        self._write_event("dataset", {
            "name": name, "n_rows": n_rows,
            "n_features": n_features, "target": target,
            "class_distribution": class_distribution,
        })

    def log_model(
        self, model_name: str, params: dict[str, Any],
        accuracy: Optional[float] = None,
    ) -> None:
        self.logger.info(
            "Model %s: params=%s, accuracy=%s",
            model_name, params, accuracy,
        )
        self._write_event("model", {
            "model_name": model_name, "params": params,
            "accuracy": accuracy,
        })

    # ── pool building ────────────────────────────────────────

    def log_pool(
        self, query_uuid: str, pool_size: int,
        is_perturbed: bool, sigma: Optional[float] = None,
    ) -> None:
        self.logger.info(
            "Pool built: query=%s, size=%d, perturbed=%s, sigma=%s",
            query_uuid, pool_size, is_perturbed, sigma,
        )
        self._write_event("pool", {
            "query_uuid": query_uuid, "pool_size": pool_size,
            "is_perturbed": is_perturbed, "sigma": sigma,
        })

    # ── evaluation ───────────────────────────────────────────

    def log_query_result(
        self, query_uuid: str, sigma: float,
        n_candidates: int, n_pareto: int,
        mean_proximity: float,
        mean_geo_instability: float,
        mean_int_instability: float,
    ) -> None:
        self.logger.info(
            "Query %s (sigma=%.4f): %d candidates, %d Pareto, "
            "prox=%.4f, geo=%.4f, int=%.4f",
            query_uuid, sigma, n_candidates, n_pareto,
            mean_proximity, mean_geo_instability,
            mean_int_instability,
        )
        self._write_event("query_result", {
            "query_uuid": query_uuid, "sigma": sigma,
            "n_candidates": n_candidates, "n_pareto": n_pareto,
            "mean_proximity": mean_proximity,
            "mean_geometric_instability": mean_geo_instability,
            "mean_intervention_instability": mean_int_instability,
        })

    def log_stability_auc(
        self, dataset: str, geometric_auc: float,
        intervention_auc: float,
    ) -> None:
        self.logger.info(
            "AUC [%s]: geometric=%.4f, intervention=%.4f",
            dataset, geometric_auc, intervention_auc,
        )
        self._write_event("stability_auc", {
            "dataset": dataset,
            "geometric_auc": geometric_auc,
            "intervention_auc": intervention_auc,
        })

    # ── general purpose ──────────────────────────────────────

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg)
        if kwargs:
            self._write_event("info", {"message": msg, **kwargs})

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg)
        self._write_event("warning", {"message": msg, **kwargs})
