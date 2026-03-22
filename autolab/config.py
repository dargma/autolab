"""Config validation with dataclasses for goal.yaml and config.yaml."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Dataclasses ──────────────────────────────────────────────────


@dataclass
class MetricSpec:
    target: float
    direction: str = "maximize"

    def __post_init__(self):
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{self.direction}'")


@dataclass
class GoalConfig:
    metrics: dict[str, MetricSpec]
    dataset: str
    stop_when: str = "all_met"
    max_iterations: int = 10

    def __post_init__(self):
        if not self.metrics:
            raise ValueError("goal.yaml must define at least one metric")


@dataclass
class CandidateConfig:
    name: str
    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintConfig:
    fps: int = 30
    device: str = "cpu"
    batch_size: int = 1
    max_latency_ms: float = 100.0


@dataclass
class TrainingConfig:
    epochs: int = 5
    lr: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "none"
    seed: int = 42
    label_smoothing: float = 0.0
    weight_decay: float = 0.0
    augmentation: bool = False
    jit: bool = False
    batch_size_train: int = 128
    batch_size_test: int = 256


@dataclass
class EvaluationConfig:
    n_runs: int = 100
    n_warmup: int = 10
    seed: int = 42


@dataclass
class SweepConfig:
    candidates: list[CandidateConfig]
    training: TrainingConfig
    constraint: ConstraintConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    parallel_workers: int = 8
    metric: str = "accuracy"
    budget: str = "600s"


@dataclass
class ProjectConfig:
    name: str
    dataset: str
    goal: GoalConfig
    sweep: SweepConfig


# ── Loaders ──────────────────────────────────────────────────────


def load_goal(path: str | Path) -> GoalConfig:
    """Load goal.yaml and return a validated GoalConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Goal file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"goal.yaml must be a YAML mapping, got {type(raw).__name__}")

    # Required keys
    if "metrics" not in raw:
        raise ValueError("goal.yaml missing required key: 'metrics'")
    if "dataset" not in raw:
        raise ValueError("goal.yaml missing required key: 'dataset'")

    metrics = {}
    for name, spec in raw["metrics"].items():
        if not isinstance(spec, dict) or "target" not in spec:
            raise ValueError(f"metric '{name}' must have a 'target' value")
        metrics[name] = MetricSpec(
            target=spec["target"],
            direction=spec.get("direction", "maximize"),
        )

    return GoalConfig(
        metrics=metrics,
        dataset=raw["dataset"],
        stop_when=raw.get("stop_when", "all_met"),
        max_iterations=raw.get("max_iterations", 10),
    )


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load config.yaml and return a validated SweepConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"config.yaml must be a YAML mapping, got {type(raw).__name__}")

    # Candidates
    search = raw.get("search", {})
    raw_candidates = search.get("candidates", raw.get("candidates"))
    if not raw_candidates:
        raise ValueError("config.yaml must define candidates (under search.candidates or candidates)")

    candidates = []
    for c in raw_candidates:
        if "name" not in c:
            raise ValueError(f"Each candidate must have a 'name': {c}")
        if "type" not in c:
            raise ValueError(f"Candidate '{c['name']}' must have a 'type'")
        kwargs = {k: v for k, v in c.items() if k not in ("name", "type")}
        candidates.append(CandidateConfig(name=c["name"], type=c["type"], kwargs=kwargs))

    # Training config
    train_raw = raw.get("train", {})
    training = TrainingConfig(
        epochs=train_raw.get("epochs", 5),
        lr=train_raw.get("lr", 0.001),
        optimizer=train_raw.get("optimizer", "adam"),
        scheduler=train_raw.get("scheduler", "none"),
        seed=train_raw.get("seed", 42),
        label_smoothing=train_raw.get("label_smoothing", 0.0),
        weight_decay=train_raw.get("weight_decay", 0.0),
        augmentation=train_raw.get("augmentation", False),
        jit=train_raw.get("jit", False),
        batch_size_train=train_raw.get("batch_size_train", 128),
        batch_size_test=train_raw.get("batch_size_test", 256),
    )

    # Constraint
    cons_raw = raw.get("constraint", {})
    constraint = ConstraintConfig(
        fps=cons_raw.get("fps", 30),
        device=cons_raw.get("device", "cpu"),
        batch_size=cons_raw.get("batch_size", 1),
        max_latency_ms=cons_raw.get("max_latency_ms", 100.0),
    )

    # Evaluation
    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        n_runs=eval_raw.get("n_runs", 100),
        n_warmup=eval_raw.get("n_warmup", 10),
        seed=eval_raw.get("seed", 42),
    )

    return SweepConfig(
        candidates=candidates,
        training=training,
        constraint=constraint,
        evaluation=evaluation,
        parallel_workers=search.get("parallel_workers", 8),
        metric=search.get("metric", "accuracy"),
        budget=search.get("budget", "600s"),
    )
