from pathlib import Path
from dataclasses import dataclass
from enum import Enum


@dataclass
class DefaultPaths:
    poolPath: Path = Path("../results/pools")
    modelsPath: Path = Path("../results/models")
    figuresPath: Path = Path("../results/figures")
    rawPath: Path = Path("../results/raw")
    tabelsPath: Path = Path("../results/tables")


class GeometricDistanceType(Enum):
    L_1 = "l1"
    L_2 = "l2"
    L_INF = "l_inf"
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"


class InterventionDistanceType(Enum):
    JACCARD_INDEX = "jaccard_index"
    DICE_SORENSEN_COEFFICIENT = "dice_sorensen_coefficient"
