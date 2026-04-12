import pytest

from src.utils.config_loader import load_yaml, load_config, load_all_dataset_configs


class TestLoadYaml:
    def test_loads_existing_yaml(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml(f)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert load_yaml(f) == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "str.yaml"
        f.write_text("x: 42\n")
        result = load_yaml(str(f))
        assert result == {"x": 42}


class TestLoadConfig:
    @pytest.fixture()
    def cfg_root(self, tmp_path, monkeypatch):
        """Create a minimal config tree and patch CONFIGS_DIR."""
        import src.utils.config_loader as mod

        monkeypatch.setattr(mod, "CONFIGS_DIR", tmp_path)

        (tmp_path / "dataset").mkdir()
        (tmp_path / "model").mkdir()
        (tmp_path / "cf_method").mkdir()
        (tmp_path / "experiment").mkdir()

        (tmp_path / "dataset" / "adult.yaml").write_text(
            "dataset:\n  name: adult-income\n"
        )
        (tmp_path / "model" / "pytorch_classifier.yaml").write_text(
            "model:\n  name: pytorch-classifier\n"
        )
        (tmp_path / "cf_method" / "dice.yaml").write_text(
            "cf_method:\n  name: dice\n"
        )
        (tmp_path / "experiment" / "robustness.yaml").write_text(
            "experiment:\n  sigmas: [0.01, 0.05]\n"
        )
        return tmp_path

    def test_dataset_only(self, cfg_root):
        cfg = load_config("adult")
        assert cfg["dataset"]["name"] == "adult-income"
        assert "model" not in cfg

    def test_dataset_and_model(self, cfg_root):
        cfg = load_config("adult", model="pytorch_classifier")
        assert cfg["dataset"]["name"] == "adult-income"
        assert cfg["model"]["name"] == "pytorch-classifier"

    def test_all_components(self, cfg_root):
        cfg = load_config(
            "adult",
            model="pytorch_classifier",
            cf_method="dice",
            experiment="robustness",
        )
        assert "dataset" in cfg
        assert "model" in cfg
        assert "cf_method" in cfg
        assert "experiment" in cfg

    def test_missing_dataset_raises(self, cfg_root):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent")


class TestLoadAllDatasetConfigs:
    @pytest.fixture()
    def cfg_root(self, tmp_path, monkeypatch):
        import src.utils.config_loader as mod

        monkeypatch.setattr(mod, "CONFIGS_DIR", tmp_path)

        (tmp_path / "dataset").mkdir()
        (tmp_path / "dataset" / "adult.yaml").write_text(
            "dataset:\n  name: adult-income\n"
        )
        (tmp_path / "dataset" / "compas.yaml").write_text(
            "dataset:\n  name: compas-recidivism\n"
        )
        return tmp_path

    def test_loads_all(self, cfg_root):
        configs = load_all_dataset_configs()
        assert len(configs) == 2
        assert "adult-income" in configs
        assert "compas-recidivism" in configs

    def test_fallback_to_stem(self, tmp_path, monkeypatch):
        import src.utils.config_loader as mod

        monkeypatch.setattr(mod, "CONFIGS_DIR", tmp_path)
        (tmp_path / "dataset").mkdir()
        (tmp_path / "dataset" / "custom.yaml").write_text("key: val\n")

        configs = load_all_dataset_configs()
        assert "custom" in configs
