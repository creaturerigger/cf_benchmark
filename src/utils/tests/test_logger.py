import json
import logging

import pytest

from src.utils.logger import get_logger, ExperimentLogger, LOGS_DIR


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_basic", log_to_file=False)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_basic"

    def test_console_handler_present(self):
        logger = get_logger("test_console", log_to_file=False)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_file_handler_created(self, tmp_path, monkeypatch):
        import src.utils.logger as mod

        monkeypatch.setattr(mod, "LOGS_DIR", tmp_path)
        logger = get_logger("test_file_handler", log_to_file=True)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in handler_types
        assert (tmp_path / "test_file_handler.log").exists()

    def test_idempotent(self):
        a = get_logger("test_idempotent", log_to_file=False)
        b = get_logger("test_idempotent", log_to_file=False)
        assert a is b
        assert len(a.handlers) == len(b.handlers)


class TestExperimentLogger:
    @pytest.fixture()
    def exp_logger(self, tmp_path, monkeypatch):
        import src.utils.logger as mod

        monkeypatch.setattr(mod, "LOGS_DIR", tmp_path)
        return ExperimentLogger("test_exp")

    def _read_events(self, logger: ExperimentLogger):
        lines = logger.jsonl_path.read_text().strip().splitlines()
        return [json.loads(line) for line in lines]

    def test_jsonl_file_created(self, exp_logger):
        exp_logger.info("hello", detail="world")
        assert exp_logger.jsonl_path.exists()

    def test_log_config(self, exp_logger):
        cfg = {"dataset": {"name": "adult"}, "model": {"layers": [20]}}
        exp_logger.log_config(cfg)
        events = self._read_events(exp_logger)
        assert len(events) == 1
        assert events[0]["event"] == "config"
        assert events[0]["config"] == cfg

    def test_log_seed(self, exp_logger):
        exp_logger.log_seed(42)
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "seed"
        assert events[0]["seed"] == 42

    def test_log_dataset(self, exp_logger):
        exp_logger.log_dataset(
            "adult", n_rows=1000, n_features=14,
            target="income", class_distribution={"<=50K": 700, ">50K": 300},
        )
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "dataset"
        assert events[0]["n_rows"] == 1000

    def test_log_model(self, exp_logger):
        exp_logger.log_model("pytorch-classifier", {"layers": [20]}, accuracy=0.85)
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "model"
        assert events[0]["accuracy"] == 0.85

    def test_log_pool(self, exp_logger):
        exp_logger.log_pool("uuid-1", pool_size=50, is_perturbed=False)
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "pool"
        assert events[0]["pool_size"] == 50
        assert events[0]["is_perturbed"] is False

    def test_log_query_result(self, exp_logger):
        exp_logger.log_query_result(
            query_uuid="uuid-1", sigma=0.05,
            n_candidates=10, n_pareto=3,
            mean_proximity=0.12,
            mean_geo_instability=0.05,
            mean_int_instability=0.08,
        )
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "query_result"
        assert events[0]["n_pareto"] == 3
        assert events[0]["mean_proximity"] == 0.12

    def test_log_stability_auc(self, exp_logger):
        exp_logger.log_stability_auc("adult", geometric_auc=0.03, intervention_auc=0.07)
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "stability_auc"
        assert events[0]["geometric_auc"] == 0.03

    def test_warning_writes_event(self, exp_logger):
        exp_logger.warning("something off", detail="extra")
        events = self._read_events(exp_logger)
        assert events[0]["event"] == "warning"
        assert events[0]["detail"] == "extra"

    def test_multiple_events_appended(self, exp_logger):
        exp_logger.log_seed(1)
        exp_logger.log_seed(2)
        events = self._read_events(exp_logger)
        assert len(events) == 2
        assert events[0]["seed"] == 1
        assert events[1]["seed"] == 2

    def test_event_has_timestamp_and_experiment_id(self, exp_logger):
        exp_logger.log_seed(42)
        events = self._read_events(exp_logger)
        assert "timestamp" in events[0]
        assert events[0]["experiment_id"] == "test_exp"
