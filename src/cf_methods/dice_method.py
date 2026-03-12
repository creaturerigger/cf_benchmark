from .base_cf_method import BaseCounterfactualGenerationMethod
from .registry import register_method


@register_method(name='dice')
class DiCEMethod(BaseCounterfactualGenerationMethod):

    def __init__(self, cfg, model, data_interface):
        super().__init__(cfg)
        self.model = model
        self.data_interface = data_interface

    def generate(self, query_instance, num_cfs: int, **kwargs):
        self.explainer = self.generate_counterfactuals(query_instance,
                                                       total_CFS=num_cfs,
                                                       **kwargs)
        return self.explainer
