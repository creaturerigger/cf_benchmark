from pathlib import Path
from dataclasses import dataclass
from enum import Enum


@dataclass
class DefaultPaths:
    poolPath: Path = Path("results/pools")
    modelsPath: Path = Path("results/models")
    figuresPath: Path = Path("results/figures")
    rawPath: Path = Path("results/raw")
    tablesPath: Path = Path("results/tables")
    logsPath: Path = Path("results/logs")

    @classmethod
    def for_method(cls, cf_method: str) -> "DefaultPaths":
        """Return paths scoped under ``results/<cf_method>/``.

        Models are kept at ``results/models/`` (shared across methods).
        """
        base = Path("results") / cf_method
        return cls(
            poolPath=base / "pools",
            modelsPath=Path("results/models"),
            figuresPath=base / "figures",
            rawPath=base / "raw",
            tablesPath=base / "tables",
            logsPath=base / "logs",
        )


class GeometricDistanceType(Enum):
    L_1 = "l1"
    L_2 = "l2"
    L_INF = "l_inf"
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"


class InterventionDistanceType(Enum):
    JACCARD_INDEX = "jaccard_index"
    DICE_SORENSEN_COEFFICIENT = "dice_sorensen_coefficient"
