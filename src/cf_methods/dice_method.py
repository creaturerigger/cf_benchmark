from dataclasses import dataclass

import pandas as pd
import dice_ml_x
from dice_ml_x.constants import BackEndTypes

from .base_cf_method import BaseCounterfactualGenerationMethod
from .registry import register_method


@dataclass
class DiCEResult:
    """Adapter so pool builder can call ``result.to_dataframe()``."""

    cfs: pd.DataFrame | None
    feature_names: list[str]

    def to_dataframe(self) -> pd.DataFrame | None:
        return self.cfs


@register_method(name='dice')
class DiCEMethod(BaseCounterfactualGenerationMethod):
    """Wraps DiCE-X for counterfactual generation with a PyTorch classifier."""

    def __init__(self, cfg: dict, model, dataframe, target_column: str,
                 continuous_features: list[str]):
        """
        Args:
            cfg: YAML config dict (from dice.yaml).
            model: trained PYTModel instance.
            dataframe: the full training DataFrame (used by DiCE-X Data).
            target_column: name of the target column in dataframe.
            continuous_features: list of continuous feature names.
        """
        super().__init__(cfg)

        self.data_interface = dice_ml_x.Data(
            dataframe=dataframe,
            continuous_features=continuous_features,
            outcome_name=target_column,
        )
        self.model_interface = dice_ml_x.Model(
            model=model,
            backend=BackEndTypes.Pytorch,
            func="ohe-min-max",
        )
        method = cfg.get("search", {}).get("algorithm", "gradient")
        dice_kwargs = {}
        if method == "gradient":
            dice_kwargs["dice_x"] = cfg.get("dice_x", {}).get("enabled", False)
        self.explainer = dice_ml_x.Dice(
            self.data_interface,
            self.model_interface,
            method=method,
            **dice_kwargs,
        )

    @property
    def encoded_categorical_feature_indices(self) -> list[list[int]]:
        return self.data_interface.get_encoded_categorical_feature_indexes()

    @property
    def encoded_continuous_feature_indices(self) -> list[int]:
        encoded_cats = {i for group in self.encoded_categorical_feature_indices for i in group}
        total = len(self.data_interface.get_encoded_feature_names())
        return [i for i in range(total) if i not in encoded_cats]

    def generate(self, query_instance, num_cfs: int, **kwargs):
        """Generate counterfactuals for query_instance.

        Args:
            query_instance: a single-row DataFrame or dict.
            num_cfs: number of counterfactuals to generate.
            **kwargs: forwarded to generate_counterfactuals (e.g.
                      desired_class, permitted_range, features_to_vary).
        Returns:
            DiCEResult wrapping the generated CFs as a feature-only DataFrame.
        """
        gen_cfg = self.cfg.get("generation", {})
        result = self.explainer.generate_counterfactuals(
            query_instance,
            total_CFs=num_cfs,
            desired_class=kwargs.pop("desired_class", gen_cfg.get("desired_class", "opposite")),
            proximity_weight=kwargs.pop("proximity_weight", gen_cfg.get("proximity_weight", 0.5)),
            diversity_weight=kwargs.pop("diversity_weight", gen_cfg.get("diversity_weight", 1.0)),
            init_near_query_instance=kwargs.pop("init_near_query_instance",
                                                gen_cfg.get("init_near_query_instance", True)),
            **kwargs,
        )

        target_col = self.data_interface.outcome_name
        feature_names = [
            c for c in self.data_interface.feature_names
        ]

        cfs_df = result.cf_examples_list[0].final_cfs_df
        if cfs_df is not None and len(cfs_df) > 0:
            cfs_df = cfs_df.drop(columns=[target_col], errors="ignore")

        return DiCEResult(cfs=cfs_df, feature_names=feature_names)
