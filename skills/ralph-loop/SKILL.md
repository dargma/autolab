---
name: ralph-loop
description: Autonomous iteration engine for neural architecture search
user_invocable: true
---

# Ralph-Loop Skill

Runs the autonomous ralph-loop iteration engine to achieve a target goal.

## Commands

### `/ralph run [project_dir]`
Execute the ralph-loop with the project's goal.yaml.
1. Load goal.yaml from project directory
2. Check current best result against target metrics
3. If goal met: generate final report
4. If not: select strategy, run experiment, loop

### `/ralph status [project_dir]`
Show current ralph-loop status:
- Current iteration number
- Best result vs target gap
- Selected strategy for next iteration
- Ralph-log.json summary

### `/ralph report [project_dir]`
Generate a final ICLR/CVPR-quality report from all ralph iterations:
- Reads all experiment results
- Generates comparison figures
- Updates PROGRESS.md with comprehensive analysis

## Usage

```bash
# Run the autonomous loop
python -m autolab.ralph projects/mnist-cpu/goal.yaml

# Generate dashboard
python -m autolab.dashboard projects/mnist-cpu/

# From Claude Code
/ralph run projects/mnist-cpu
```

## Architecture

```
Load Goal → Check Best → Goal Met? → Yes → Report → Done
                              ↓ No
                    Analyze Gap → Select Strategy → Run Experiment → Loop
```

Strategies: ArchitectureSearch, HyperparameterTuning, TrainingExtension, Augmentation, Regularization, ModelCompression
