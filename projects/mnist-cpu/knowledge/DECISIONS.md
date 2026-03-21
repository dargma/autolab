# Research Decisions Log

> Record direction changes and key decisions in chronological order. Always read this file first at session start.

<!-- Example:
### 2025-03-17: Switched from LSR to IIG-based reward
- Reason: LSR alone hit a ceiling on blind gap improvement
- Before: LSR-only GRPO
- After: IIG = log-prob ratio(real/black), λ auto-calibrate
- Impact: Full reward function replacement, exp-001~003 become baselines
-->

### 2026-03-17: Add CPU-constrained MNIST architecture search
- Reason: Validate auto-research system with a lightweight, fast-turnaround task
- Before: No active experiment
- After: Find optimal MNIST classifier under 10fps CPU constraint
- Impact: First use of multi-CPU sweep protocol

### 2026-03-21: Restructure into reusable autolab package + project scaffold
- Reason: Code was duplicated between exp-001 and exp-002; no shared framework, no autonomous loop, no dashboard
- Before: Standalone sweep.py per experiment, no package structure, no ralph-loop
- After: autolab/ Python package (models, data, sweep, ralph, dashboard, figures, knowledge), projects/mnist-cpu/ scaffold with goal.yaml, ralph-loop autonomous iteration engine
- Impact: All future experiments use shared framework. Existing exp-001/002 migrated to projects/mnist-cpu/
