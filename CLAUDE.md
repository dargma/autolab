# Autolab — Autonomous AI Research Framework

> **Version 0.2.0** | Framework for autonomous neural network research, architecture search, and experiment management.
> Clone → scaffold → define goal → run → iterate. Works with Claude Code or standalone.

---

## Quick Start (New Research Project)

```bash
# 1. Scaffold a new project
python -m autolab new cifar10-edge

# 2. Edit targets
#    projects/cifar10-edge/goal.yaml    → accuracy, latency, etc.
#    projects/cifar10-edge/config.yaml  → model candidates, training config

# 3. (Optional) Add custom models
#    projects/cifar10-edge/models.py    → @register("my_arch") class MyArch(nn.Module): ...

# 4. Run architecture sweep
python -m autolab sweep projects/cifar10-edge

# 5. Check results
#    experiments/TRACKER.md, dashboard.html, reports/figures/

# 6. Autonomous iteration (goal-driven loop)
python -m autolab ralph projects/cifar10-edge
```

See `GETTING_STARTED.md` for the full walkthrough.

---

## 1. Framework Architecture

```
autolab/                          # ★ REUSABLE PACKAGE (pip-installable)
├── __init__.py                   # Public API: register, build_model, get_loaders, etc.
├── __main__.py                   # CLI: python -m autolab {new,sweep,ralph,bench,dashboard,report}
├── models.py                     # Model registry with @register decorator
├── data.py                       # Dataset factory (MNIST, FashionMNIST, CIFAR10, ...)
├── config.py                     # Config validation (GoalConfig, SweepConfig, CandidateConfig)
├── sweep.py                      # Parallel sweep engine (memory-aware, checkpoint, resume)
├── ralph.py                      # Autonomous iteration engine (goal → gap → strategy → run)
├── inference.py                  # Unified C inference API (ternary/quantized models)
├── distill.py                    # Knowledge distillation (teacher → student)
├── scaffold.py                   # Project scaffolding (creates directory structure)
├── dashboard.py                  # Single-file HTML dashboard generator
├── figures.py                    # CVPR-quality matplotlib figure utilities
├── knowledge.py                  # Markdown parsers (TRACKER, REGISTRY, DECISIONS)
├── safety.py                     # Disk guard + resource checks
├── csrc/                         # C inference kernels (ternary, quantized)
├── templates/                    # Project templates (copied by scaffold)
├── plugins/                      # Plugin system (lifecycle hooks)
└── tests/                        # pytest test suite

projects/                         # PROJECT-SPECIFIC (one per research task)
└── {project-name}/
    ├── CLAUDE.md                 # Project-specific instructions
    ├── goal.yaml                 # ★ What to achieve (metrics + targets)
    ├── config.yaml               # ★ How to search (candidates + training)
    ├── models.py                 # Optional: project-specific architectures
    ├── knowledge/
    │   ├── REGISTRY.md           # Cumulative knowledge (append-only)
    │   └── DECISIONS.md          # Direction changes + rationale
    ├── experiments/
    │   ├── TRACKER.md            # Experiment status matrix
    │   └── exp-{NNN}-{name}/     # Individual experiments
    │       ├── config.yaml
    │       ├── run.sh
    │       ├── REPORT.md
    │       └── results/
    │           ├── checkpoints/  # Model .pt files (auto-saved)
    │           ├── figures/      # Per-experiment figures (auto-generated)
    │           ├── sweep-*.csv
    │           └── train.log
    ├── reports/
    │   ├── PROGRESS.md           # Cumulative CVPR-quality report
    │   ├── REPORT-obsidian.md    # Obsidian-formatted report (optional)
    │   └── figures/
    └── dashboard.html

skills/                           # Claude Code skills (reusable)
├── ralph-loop/SKILL.md           # /ralph {run,status,report}
├── auto-research/SKILL.md        # /research {survey,ideate,run,synthesize}
└── exp-harness/SKILL.md          # /experiment {new,run,report}
```

---

## 2. Core Concepts

### goal.yaml — What You Want
```yaml
metrics:
  accuracy: {target: 0.95, direction: maximize}
  avg_latency_ms: {target: 10.0, direction: minimize}
dataset: MNIST           # Any registered dataset
stop_when: all_met        # or: any_met, max_iterations
max_iterations: 10
```

### config.yaml — How to Search
```yaml
training:
  epochs: 10
  optimizer: AdamW
  lr: 0.003
  scheduler: cosine
  label_smoothing: 0.05
  patience: 5             # early stopping

constraint:
  avg_latency_ms: 10.0
  device: cpu

candidates:
  - {name: my-cnn, type: cnn, channels: [1,16,32], fc: [256,10]}
  - {name: my-ternary, type: ternary_cnn, channels: [1,8,16], fc: [128,10]}
```

### @register — Model Registry
```python
from autolab.models import register
import torch.nn as nn

@register("my_custom_arch")
class MyArch(nn.Module):
    def __init__(self, channels, fc, input_size=28, **_kw):
        ...
    def forward(self, x):
        ...
```

Models registered in `autolab/models.py` are available everywhere. Project-specific models go in `projects/{name}/models.py` — import them in config or sweep script.

---

## 3. Knowledge Management Rules

### REGISTRY.md — Single Source of Truth
- **Never delete entries.** Append and revise only.
- Sections: `## Established Facts`, `## Hypotheses`, `## Rejected Ideas`, `## Open Questions`
- After each paper: add 3-line summary to relevant section
- After each experiment: update facts/hypotheses with results

### DECISIONS.md — Direction Change History
- **Must** be updated whenever direction or plans change
- Format: `### YYYY-MM-DD: {title}` + reason + before→after + impact scope
- This file is long-term memory — always read it at session start

### TRACKER.md — Experiment Matrix
- Table: `| # | Name | Status | Key Metric | Date | Notes |`
- Status: 🔵planned → 🟡running → 🟢done → 🔴failed
- Auto-updated by sweep engine and ralph-loop

### PROGRESS.md — Cumulative Report (CVPR Quality)
- Every claim must have a figure or table
- Baselines always included in comparisons
- One ablation variable at a time
- Axis labels, legends, captions mandatory
- **Append to existing report** — never create a new file

---

## 4. Session Start Routine

At every new session, **always**:
1. Read `DECISIONS.md` (recent direction changes)
2. Read `REGISTRY.md` (current knowledge state)
3. Read `TRACKER.md` (in-progress experiments)
4. Check disk usage (`autolab.safety.check_disk()`)
5. Then begin work

---

## 5. Disk Safety Guard

**Before every file write or experiment launch:**
```bash
USAGE=$(df -h . | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$USAGE" -ge 95 ]; then
  echo "DISK 95% — All operations halted. Cleanup required."
  exit 1
fi
```
Enforced automatically by `safety.check_disk()` in sweep and ralph.

---

## 6. Research Workflow (4 Phases)

### Phase 1: Survey
1. Read papers in `papers/` → extract key contribution, method, applicability, limitations
2. Update REGISTRY.md sections
3. Cross-analyze: does new paper support/refute existing hypotheses?

### Phase 2: Ideation

**Never produce only safe variations.** Required:

| # | Name | Novelty | Feasibility | Description |
|---|------|---------|-------------|-------------|
| A | ... | ★★★ | ★★☆ | Bold idea |
| B | ... | ★☆☆ | ★★★ | Safe improvement |

Checklist:
- [ ] At least 2 safe improvements + 2 bold fusion ideas + 1 cross-domain transplant
- [ ] Pros/cons comparison across all candidates
- [ ] Explicitly tried "What if we combine A+B?"
- [ ] At least 1 idea from another field (CV, NLP, RL, biology, physics)
- [ ] Recorded rejections with reasons in REGISTRY

### Phase 3: Experiment
1. Register in TRACKER.md (auto via `python -m autolab sweep`)
2. Write config.yaml (fully reproducible)
3. Execute → checkpoints auto-saved, metrics logged
4. Write REPORT.md (CVPR-level: objective, setup, results table+figure, analysis, next steps)
5. Update TRACKER.md status
6. Figures auto-generated + dashboard refreshed

### Phase 4: Synthesis
- Review TRACKER.md → update PROGRESS.md
- Generate best-vs-baseline comparison figure
- Propose next steps → update REGISTRY.md `## Open Questions`

---

## 7. Experiment REPORT.md Template

```markdown
# Exp-{NNN}: {name}
## Objective       — 1 line: the hypothesis this tests
## Setup           — config summary + differences from baseline only
## Results         — table + figure mandatory
## Analysis        — why this result? support or reject hypothesis
## Next Steps      — actions derived from this result
```

### Figure Rules
- matplotlib/seaborn, save to `reports/figures/`
- Axis labels, legends, captions mandatory
- Baseline always shown as dashed line
- Colors: colorblind-safe palettes (seaborn)
- 300 DPI, prefer SVG
- Naming: `fig-{exp-number}-{content}.svg`

---

## 8. Ralph-Loop (Autonomous Iteration)

```
Load Goal (goal.yaml)
    ↓
Check Best Result (scan all experiments)
    ↓
Goal Met? → Yes → Generate Final Report → Done
    ↓ No
Analyze Gap (which metrics are short?)
    ↓
Select Strategy:
  - ArchitectureSearch    (first run or accuracy gap > 5pp)
  - HyperparameterTuning  (accuracy gap < 5pp)
  - TrainingExtension      (loss still decreasing, load checkpoint)
  - ModelCompression       (need latency improvement → ternary/quantize)
  - KnowledgeDistillation  (accuracy close, have a good teacher)
  - Refinement             (augmentation + regularization)
    ↓
Run Experiment → Log Results → Loop Back
```

Each iteration logged to `ralph-log.json` with full reasoning trace.

---

## 9. Dashboard (dashboard.html)

Auto-generated after each experiment. Required panels:
1. **Experiment status matrix** — from TRACKER.md (colored by status)
2. **Metric trend chart** — X: experiment, Y: key metric, target as horizontal line
3. **Current best result** — best experiment, metric, improvement over baseline
4. **Resource status** — disk usage, experiment count
5. **Research timeline** — decisions + experiment completions

Single HTML file, Chart.js CDN, opens via `file://`.

---

## 10. Multi-CPU Architecture Search

```yaml
constraint:
  avg_latency_ms: 10.0
  device: cpu
  batch_size: 1
search:
  parallel_workers: auto    # auto = memory-aware scaling
  candidates: [...]
  metric: accuracy
```

Flow: generate candidates → parallel train (memory-aware pool) → auto-checkpoint → constraint filter → rank → CSV + figures → TRACKER + dashboard.

Features:
- **Memory-aware workers**: auto-scales pool based on available RAM
- **Checkpointing**: best model saved per candidate to `results/checkpoints/`
- **Resume**: skips candidates already in live CSV (survives OOM)
- **Early stopping**: patience-based (default 5 epochs)

---

## 11. Ternary / Quantized Inference

For models with ternary weights `{-1, 0, +1}`:

```python
from autolab.inference import TernaryInference
engine = TernaryInference(version="v3")
result = engine.benchmark(model, n_warmup=200, n_runs=2000)
print(f"Avg: {result['avg_ms']:.3f}ms")
```

The C engine uses:
- **Zero multiplications**: pure add/sub with int16 activations
- **Zero-skipping**: skip ~35-45% of zero weights
- **Fused BN+ReLU**: single pass per channel
- **Auto-compile**: builds .so from C source on first use

---

## 12. Skills Reference

| Skill | Commands | Purpose |
|-------|----------|---------|
| `/ralph` | `run`, `status`, `report` | Autonomous goal-driven iteration |
| `/research` | `survey`, `ideate`, `run`, `synthesize` | Full research workflow |
| `/experiment` | `new <name>`, `run <dir>`, `report <dir>` | Single experiment management |

---

## 13. Adding New Components

### New Dataset
Add to `autolab/data.py`:
```python
DATASETS["MyDataset"] = (MyDatasetClass, (mean,), (std,), channels, size, num_classes)
```

### New Model Architecture
Add to `autolab/models.py` or project's `models.py`:
```python
@register("my_model")
class MyModel(nn.Module):
    def __init__(self, channels, fc, input_size=28, **_kw): ...
```

### New Ralph Strategy
Add to `autolab/ralph.py` in `_run_strategy()`:
```python
elif strategy == "MyStrategy":
    # Create experiment, run training, return results
```

### New Plugin
Extend `autolab/plugins/base.py`:
```python
class MyPlugin(AutolabPlugin):
    def on_experiment_end(self, exp_dir, results): ...
```
