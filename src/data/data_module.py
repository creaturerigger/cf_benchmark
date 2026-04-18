from .registry import load_dataset, register_dataset
from .datasets.adult import load_adult
from .datasets.compas import load_compas
from .datasets.lending import load_lending
from .datasets.german import load_german
from .datasets.heloc import load_heloc
from .datasets.diabetes import load_diabetes

__all__ = ["load_dataset", "register_dataset"]
