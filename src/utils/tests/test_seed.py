import os
import random

import numpy as np
import torch

from src.utils.seed import set_seed


class TestSetSeed:
    def test_python_random_deterministic(self):
        set_seed(123)
        a = [random.random() for _ in range(10)]
        set_seed(123)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_deterministic(self):
        set_seed(123)
        a = np.random.rand(10)
        set_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self):
        set_seed(123)
        a = torch.rand(10)
        set_seed(123)
        b = torch.rand(10)
        assert torch.equal(a, b)

    def test_pythonhashseed_env_var(self):
        set_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_cudnn_flags(self):
        set_seed(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_default_seed(self):
        set_seed()
        a = torch.rand(5)
        set_seed()
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.rand(10)
        set_seed(2)
        b = torch.rand(10)
        assert not torch.equal(a, b)
