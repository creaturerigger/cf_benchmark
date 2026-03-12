from abc import ABC, abstractmethod


class BaseCounterfactualGenerationMethod(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def generate(self, query_instance, num_cfs: int, *args, **kwargs):
        pass
