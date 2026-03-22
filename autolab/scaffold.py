"""Project scaffolding — create new autolab project directories."""

from pathlib import Path


def create_project(name: str, base_dir: str | Path = "projects") -> Path:
    """Create a full project directory structure under base_dir/<name>.

    Returns the path to the created project directory.
    """
    base_dir = Path(base_dir)
    project_dir = base_dir / name
    if project_dir.exists():
        raise FileExistsError(f"Project already exists: {project_dir}")

    # Create directories
    (project_dir / "knowledge").mkdir(parents=True)
    (project_dir / "experiments").mkdir(parents=True)
    (project_dir / "reports" / "figures").mkdir(parents=True)
    (project_dir / "papers").mkdir(parents=True)

    # CLAUDE.md — project-specific instructions
    (project_dir / "CLAUDE.md").write_text(f"""\
# {name} — Project Instructions

> See the root autolab CLAUDE.md for the full research workflow.
> Add project-specific rules below.

## Project Goal
<!-- Describe the research objective for this project -->

## Dataset
<!-- Which dataset(s) are used? -->

## Constraints
<!-- Latency, model size, hardware limits, etc. -->

## Notes
<!-- Any project-specific conventions or decisions -->
""")

    # goal.yaml
    (project_dir / "goal.yaml").write_text("""\
# Goal definition — what does "done" look like?
# Each metric needs: target (number) and direction (maximize/minimize)
metrics:
  accuracy: {target: 0.95, direction: maximize}
  # avg_latency_ms: {target: 1.0, direction: minimize}
dataset: MNIST
stop_when: all_met        # all_met | any_met
max_iterations: 10
""")

    # config.yaml
    (project_dir / "config.yaml").write_text("""\
# Sweep configuration — architecture search setup
task: arch_search
dataset: MNIST

constraint:
  fps: 30
  device: cpu
  batch_size: 1
  max_latency_ms: 1.0

search:
  parallel_workers: 4
  metric: accuracy
  budget: 300s
  candidates:
    # Add model candidates here. Each needs: name, type, plus model kwargs.
    # Available types: fc, cnn, cnn_bn, residual_cnn, squeeze_excite_cnn,
    #                  ternary_cnn, ternary_hybrid_cnn, depthwise
    - name: baseline-fc
      type: fc
      layers: [784, 128, 10]
    - name: baseline-cnn
      type: cnn
      channels: [1, 16, 32]
      fc: [256, 10]

train:
  epochs: 5
  lr: 0.001
  optimizer: adam          # adam | adamw | sgd
  scheduler: none          # none | cosine | onecycle
  seed: 42
  # label_smoothing: 0.0
  # weight_decay: 0.0
  # augmentation: false
  # jit: false
""")

    # knowledge/REGISTRY.md
    (project_dir / "knowledge" / "REGISTRY.md").write_text(f"""\
# {name} — Knowledge Registry

## Established Facts

## Hypotheses

## Rejected Ideas

## Open Questions
""")

    # knowledge/DECISIONS.md
    (project_dir / "knowledge" / "DECISIONS.md").write_text(f"""\
# {name} — Decision History
""")

    # experiments/TRACKER.md
    (project_dir / "experiments" / "TRACKER.md").write_text("""\
# Experiment Tracker

| # | Name | Status | Key Metric | Date | Notes |
|---|------|--------|------------|------|-------|
""")

    # reports/PROGRESS.md
    (project_dir / "reports" / "PROGRESS.md").write_text(f"""\
# {name} — Research Progress

<!-- Append results here after each experiment. Never overwrite. -->
""")

    # models.py — optional project-specific models
    (project_dir / "models.py").write_text("""\
\"\"\"Project-specific model definitions.

Register custom models here using the autolab model registry.
These will be available in config.yaml via their registered type name.

Example:
    from autolab.models import register

    @register("my_custom_model")
    class MyModel(nn.Module):
        def __init__(self, ...):
            ...
\"\"\"
""")

    print(f"Created project: {project_dir}")
    return project_dir
