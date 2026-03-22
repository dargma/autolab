"""Shared fixtures for autolab tests."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory structure with stub files."""
    proj = tmp_path / "test-project"
    proj.mkdir()

    # knowledge/
    knowledge = proj / "knowledge"
    knowledge.mkdir()

    (knowledge / "REGISTRY.md").write_text(
        "# Registry\n\n"
        "## Established Facts\n"
        "- Baseline accuracy is 95%\n\n"
        "## Hypotheses\n\n"
        "## Rejected Ideas\n\n"
        "## Open Questions\n"
    )

    (knowledge / "DECISIONS.md").write_text(
        "# Decisions\n\n"
        "### 2026-01-01: Initial direction\n"
        "- Reason: Starting project\n"
        "- Before: nothing\n"
        "- After: MNIST baseline\n"
        "- Impact: full scope\n"
    )

    # experiments/
    experiments = proj / "experiments"
    experiments.mkdir()

    (experiments / "TRACKER.md").write_text(
        "# Experiment Tracker\n\n"
        "| # | Name | Status | Key Metric | Date | Notes |\n"
        "|---|------|--------|------------|------|-------|\n"
        "| 001 | baseline | done | 95.00% | 2026-01-01 | initial run |\n"
    )

    # goal.yaml
    (proj / "goal.yaml").write_text(
        "dataset: MNIST\n"
        "metrics:\n"
        "  accuracy:\n"
        "    target: 99.0\n"
        "    direction: maximize\n"
        "stop_when: all_met\n"
        "max_iterations: 5\n"
    )

    # config.yaml
    (proj / "config.yaml").write_text(
        "search:\n"
        "  parallel_workers: 2\n"
        "  metric: accuracy\n"
        "  budget: 60s\n"
        "  candidates:\n"
        "    - name: tiny-fc\n"
        "      type: fc\n"
        "      layers: [784, 64, 10]\n"
        "constraint:\n"
        "  fps: 60\n"
        "  device: cpu\n"
        "train:\n"
        "  epochs: 1\n"
        "  lr: 0.001\n"
    )

    return proj
