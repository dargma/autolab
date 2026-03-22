"""Tests for autolab.config — goal and sweep config loading/validation."""

import pytest

from autolab.config import load_goal, load_sweep_config, GoalConfig, SweepConfig


class TestLoadGoal:
    """Test load_goal with valid and invalid YAML files."""

    def test_valid_goal(self, tmp_path):
        goal_file = tmp_path / "goal.yaml"
        goal_file.write_text(
            "dataset: MNIST\n"
            "metrics:\n"
            "  accuracy:\n"
            "    target: 99.0\n"
            "    direction: maximize\n"
            "stop_when: all_met\n"
            "max_iterations: 5\n"
        )
        goal = load_goal(goal_file)
        assert isinstance(goal, GoalConfig)
        assert goal.dataset == "MNIST"
        assert "accuracy" in goal.metrics
        assert goal.metrics["accuracy"].target == 99.0
        assert goal.max_iterations == 5

    def test_missing_metrics_raises(self, tmp_path):
        goal_file = tmp_path / "goal.yaml"
        goal_file.write_text("dataset: MNIST\n")
        with pytest.raises(ValueError, match="missing required key.*metrics"):
            load_goal(goal_file)

    def test_missing_dataset_raises(self, tmp_path):
        goal_file = tmp_path / "goal.yaml"
        goal_file.write_text(
            "metrics:\n"
            "  accuracy:\n"
            "    target: 99.0\n"
        )
        with pytest.raises(ValueError, match="missing required key.*dataset"):
            load_goal(goal_file)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_goal(tmp_path / "nonexistent.yaml")


class TestLoadSweepConfig:
    """Test load_sweep_config with valid config."""

    def test_valid_sweep_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "search:\n"
            "  parallel_workers: 4\n"
            "  metric: accuracy\n"
            "  budget: 120s\n"
            "  candidates:\n"
            "    - name: tiny-fc\n"
            "      type: fc\n"
            "      layers: [784, 64, 10]\n"
            "    - name: small-cnn\n"
            "      type: cnn\n"
            "      channels: [1, 16]\n"
            "      fc: [128, 10]\n"
            "constraint:\n"
            "  fps: 60\n"
            "  device: cpu\n"
            "train:\n"
            "  epochs: 2\n"
            "  lr: 0.01\n"
        )
        cfg = load_sweep_config(config_file)
        assert isinstance(cfg, SweepConfig)
        assert len(cfg.candidates) == 2
        assert cfg.candidates[0].name == "tiny-fc"
        assert cfg.candidates[1].type == "cnn"
        assert cfg.training.epochs == 2
        assert cfg.training.lr == 0.01
        assert cfg.constraint.fps == 60
        assert cfg.parallel_workers == 4

    def test_missing_candidates_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "search:\n"
            "  metric: accuracy\n"
            "train:\n"
            "  epochs: 1\n"
        )
        with pytest.raises(ValueError, match="candidates"):
            load_sweep_config(config_file)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sweep_config(tmp_path / "nonexistent.yaml")
