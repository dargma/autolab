---
name: exp-harness
description: Single experiment lifecycle — create, run, report
user_invocable: true
---

# Experiment Harness Skill

Manages the lifecycle of individual experiments: creation, execution, and reporting.

## Commands

### `/experiment new <name> [project_dir]`
Create a new experiment directory and register it.
1. Disk safety check
2. Read TRACKER.md, assign next experiment number
3. Create `experiments/exp-{NNN}-{name}/`
4. Copy config.yaml from project root as starting point
5. Add row to TRACKER.md with 🔵planned status
6. Output: experiment directory path and number

### `/experiment run <exp-dir> [project_dir]`
Execute an experiment.
1. Disk safety check
2. Read `exp-dir/config.yaml`
3. Generate `exp-dir/run.sh` (fully reproducible)
4. Update TRACKER.md status to 🟡running
5. Execute: run sweep or single training
6. Save results to `exp-dir/results/`
   - `sweep-{timestamp}.csv` or `metrics.csv`
   - `summary.json`
   - `train.log`
7. Update TRACKER.md:
   - 🟢done + key metric if successful
   - 🔴failed + error if not
8. Output: summary of results

### `/experiment report <exp-dir> [project_dir]`
Generate a CVPR-quality report for a completed experiment.
1. Read `exp-dir/results/` (CSV, JSON, logs)
2. Compute key metrics and delta vs baseline (exp-001)
3. Generate figures → `reports/figures/fig-{NNN}-{content}.svg`
4. Write `exp-dir/REPORT.md`:
   ```
   # Exp-{NNN}: {name}
   ## Objective       — hypothesis this tests
   ## Setup           — config diff from baseline
   ## Results         — table + figure (mandatory)
   ## Analysis        — why? support or reject hypothesis
   ## Next Steps      — derived actions
   ```
5. Append summary to `reports/PROGRESS.md`
6. Refresh dashboard.html

## Usage

```bash
# From Claude Code
/experiment new cifar-augmented projects/cifar-fast
/experiment run experiments/exp-003-cifar-augmented projects/cifar-fast
/experiment report experiments/exp-003-cifar-augmented projects/cifar-fast
```

## Lifecycle

```
/experiment new → 🔵planned → /experiment run → 🟡running → 🟢done → /experiment report
                                                           → 🔴failed
```

## Figure Requirements
- matplotlib/seaborn
- Axis labels, legends, captions mandatory
- Baseline shown as dashed line
- Colorblind-safe palettes
- 300 DPI, prefer SVG
- Save to `reports/figures/fig-{NNN}-{content}.svg`
