# Getting Started with Autolab

Autolab is an autonomous AI research framework. It manages the full loop: define a goal, search architectures, run experiments, track results, iterate.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (`pip install torch torchvision`)
- pyyaml (`pip install pyyaml`)
- Optional: `psutil` (memory-aware worker scaling), `matplotlib` / `seaborn` (figures)

```bash
pip install torch torchvision pyyaml psutil matplotlib seaborn
```

## 1. Create a New Project

```bash
python -m autolab new my-project
```

This scaffolds:

```
projects/my-project/
├── CLAUDE.md              # Project instructions for Claude Code
├── goal.yaml              # What you're trying to achieve
├── config.yaml            # Search space + training config
├── models.py              # Custom model definitions (optional)
├── knowledge/
│   ├── REGISTRY.md        # Accumulated knowledge
│   └── DECISIONS.md       # Direction changes
├── experiments/
│   └── TRACKER.md         # Experiment status matrix
├── reports/
│   ├── PROGRESS.md        # Cumulative progress report
│   └── figures/
└── papers/                # Drop PDFs here for survey phase
```

## 2. Define Your Goal

Edit `projects/my-project/goal.yaml`:

```yaml
metrics:
  accuracy: {target: 0.95, direction: maximize}
  avg_latency_ms: {target: 5.0, direction: minimize}
dataset: CIFAR10
stop_when: all_met
max_iterations: 10
```

This tells autolab: "Find a model that hits 95% accuracy on CIFAR-10 with under 5ms latency. Stop when both are met or after 10 iterations."

## 3. Define Your Search Space

Edit `projects/my-project/config.yaml`:

```yaml
task: cifar10_arch_search
dataset: CIFAR10
constraint:
  fps: 200
  device: cpu
  batch_size: 1
  max_latency_ms: 5.0
search:
  parallel_workers: 4
  metric: accuracy
  budget: 600s
  candidates:
    - name: small-cnn
      type: cnn
      channels: [3, 16, 32]
      fc: [128, 10]
    - name: big-cnn
      type: cnn
      channels: [3, 32, 64]
      fc: [512, 10]
    - name: residual-cnn
      type: residual_cnn
      channels: [3, 16, 32]
      fc: [256, 10]
    - name: se-cnn
      type: squeeze_excite_cnn
      channels: [3, 16, 32]
      fc: [256, 10]
train:
  epochs: 10
  lr: 0.001
  optimizer: adam
  seed: 42
```

## 4. (Optional) Add Custom Models

Edit `projects/my-project/models.py` to register project-specific architectures:

```python
from autolab.models import register
import torch.nn as nn

@register("my_custom_net")
class MyCustomNet(nn.Module):
    def __init__(self, channels, fc, input_size=32, **_kw):
        super().__init__()
        # your architecture here
        ...

    def forward(self, x):
        ...
```

Then reference it in config.yaml:

```yaml
candidates:
  - name: custom-v1
    type: my_custom_net
    channels: [3, 32, 64]
    fc: [256, 10]
```

## 5. Run a Sweep

```bash
python -m autolab sweep projects/my-project
```

This runs all candidates in parallel on CPU, measures accuracy and latency, and saves results to `experiments/current/results/`.

Output: `sweep-{timestamp}.csv` with columns: name, params, accuracy, avg_latency_ms, p99_latency_ms, train_time_s, meets_constraint, status.

## 6. Check Results

- **CSV**: `projects/my-project/experiments/current/results/sweep-*.csv`
- **JSON**: `projects/my-project/experiments/current/results/summary.json`
- **Dashboard**: `python -m autolab dashboard projects/my-project` (generates `dashboard.html`)
- **TRACKER.md**: Updated automatically with experiment status

## 7. Iterate with Ralph

Ralph is the autonomous iteration engine. It reads your goal, checks current best results, picks a strategy, and runs the next experiment automatically.

```bash
python -m autolab ralph projects/my-project
```

Ralph strategies: ArchitectureSearch, HyperparameterTuning, TrainingExtension, Augmentation, Regularization, ModelCompression.

## 8. Use with Claude Code

Autolab works best with Claude Code. The skills system provides structured workflows:

```
/ralph run projects/my-project      # Autonomous iteration loop
/ralph status projects/my-project   # Check progress vs goal
/research survey                    # Read papers, update knowledge
/research ideate                    # Generate experiment ideas
/experiment new cifar-baseline      # Create new experiment
/experiment run exp-001-cifar-baseline
```

## Concrete Example: CIFAR-10 Under 5ms

```bash
# 1. Scaffold
python -m autolab new cifar-fast

# 2. Set goal
cat > projects/cifar-fast/goal.yaml << 'EOF'
metrics:
  accuracy: {target: 0.95, direction: maximize}
  avg_latency_ms: {target: 5.0, direction: minimize}
dataset: CIFAR10
stop_when: all_met
max_iterations: 10
EOF

# 3. Define candidates in config.yaml (see step 3 above)

# 4. Run
python -m autolab sweep projects/cifar-fast

# 5. Check best result
cat projects/cifar-fast/experiments/current/results/summary.json | python -m json.tool

# 6. If goal not met, iterate
python -m autolab ralph projects/cifar-fast
```

## Built-in Model Types

| Type | Name | Description |
|------|------|-------------|
| `fc` | Fully-connected | Simple MLP |
| `cnn` | CNN | Conv + MaxPool + FC |
| `cnn_bn` | CNN + BatchNorm | Adds BatchNorm after each conv |
| `residual_cnn` | Residual CNN | Skip connections (ResNet-style) |
| `squeeze_excite_cnn` | SE-CNN | Channel attention (SENet) |
| `depthwise` | Depthwise CNN | MobileNet-style depthwise separable |
| `ternary_cnn` | Ternary CNN | Ternary weight quantization |
| `ternary_hybrid_cnn` | Ternary Hybrid | Ternary convs + full-precision FC |

## Supported Datasets

MNIST, FashionMNIST, CIFAR10, CIFAR100 (auto-downloaded via torchvision).

## Training Options

```yaml
train:
  epochs: 10
  lr: 0.001
  optimizer: adam       # adam, adamw, sgd
  scheduler: cosine     # none, cosine, onecycle
  label_smoothing: 0.1  # 0.0 to disable
  augmentation: true    # random affine + erasing
  jit: true             # TorchScript for faster inference
  weight_decay: 0.0001
  seed: 42
```

## Project Structure After Experiments

```
projects/cifar-fast/
├── experiments/
│   ├── TRACKER.md
│   ├── exp-001-baseline-sweep/
│   │   ├── config.yaml
│   │   ├── run.sh
│   │   ├── results/
│   │   │   ├── sweep-20260322-143000.csv
│   │   │   └── summary.json
│   │   └── REPORT.md
│   └── exp-002-augmented-sweep/
│       └── ...
├── reports/
│   ├── PROGRESS.md
│   └── figures/
│       └── fig-001-accuracy-comparison.svg
└── dashboard.html
```
