from pathlib import Path

from src.utils.constants import (
    DefaultPaths,
    GeometricDistanceType,
    InterventionDistanceType,
)


class TestDefaultPaths:
    def test_default_values(self):
        paths = DefaultPaths()
        assert paths.poolPath == Path("results/pools")
        assert paths.modelsPath == Path("results/models")
        assert paths.figuresPath == Path("results/figures")
        assert paths.rawPath == Path("results/raw")
        assert paths.tablesPath == Path("results/tables")
        assert paths.logsPath == Path("results/logs")

    def test_custom_values(self):
        paths = DefaultPaths(poolPath=Path("/tmp/pools"))
        assert paths.poolPath == Path("/tmp/pools")
        assert paths.modelsPath == Path("results/models")

    def test_for_method(self):
        paths = DefaultPaths.for_method("dice")
        assert paths.poolPath == Path("results/dice/pools")
        assert paths.figuresPath == Path("results/dice/figures")
        assert paths.rawPath == Path("results/dice/raw")
        assert paths.tablesPath == Path("results/dice/tables")
        assert paths.logsPath == Path("results/dice/logs")
        # models stays universal
        assert paths.modelsPath == Path("results/models")


class TestGeometricDistanceType:
    def test_members(self):
        assert GeometricDistanceType.L_1.value == "l1"
        assert GeometricDistanceType.L_2.value == "l2"
        assert GeometricDistanceType.L_INF.value == "l_inf"
        assert GeometricDistanceType.COSINE.value == "cosine"
        assert GeometricDistanceType.MAHALANOBIS.value == "mahalanobis"

    def test_member_count(self):
        assert len(GeometricDistanceType) == 5

    def test_lookup_by_value(self):
        assert GeometricDistanceType("l1") is GeometricDistanceType.L_1


class TestInterventionDistanceType:
    def test_members(self):
        assert InterventionDistanceType.JACCARD_INDEX.value == "jaccard_index"
        assert (
            InterventionDistanceType.DICE_SORENSEN_COEFFICIENT.value
            == "dice_sorensen_coefficient"
        )

    def test_member_count(self):
        assert len(InterventionDistanceType) == 2

    def test_lookup_by_value(self):
        assert (
            InterventionDistanceType("jaccard_index")
            is InterventionDistanceType.JACCARD_INDEX
        )
