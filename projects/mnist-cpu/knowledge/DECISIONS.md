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

### 2026-03-21: Tighten goal to 98.9% accuracy + 0.50ms latency
- Reason: Previous goal (95%/33.3ms) was trivially met. User wants to push Pareto frontier.
- Before: target accuracy=0.95, latency<=33.3ms (met by all 8 candidates)
- After: target accuracy=0.989, latency<=0.50ms (no current model meets both simultaneously)
- Impact: Need new sweep with bigger CNN variants (big-cnn class) + 10 epochs training. big-cnn already hits 0.40ms but needs +0.09pp accuracy; tiny-cnn has accuracy but 0.89ms latency.

### 2026-03-21: Adopt ternary quantization with custom C inference engine
- Reason: PyTorch inference overhead makes sub-0.50ms impossible even for tiny models. Ternary weights {-1,0,+1} eliminate all multiplications — matmul becomes pure add/sub in hardware.
- Before: PyTorch inference with float32 weights, best latency = 0.40ms (big-cnn 813K params)
- After: Custom C engine (ternary_v3.c) with int16 fixed-point activations, int8 ternary weights, branchless add/sub accumulation, fused BN+ReLU
- Impact: ternary_cnn [8,16]+FC128 (103K params) achieves 0.485ms on C engine. Accuracy 99.10% (16 epochs, STE training). Both targets met simultaneously.

### 2026-03-22: Plan knowledge distillation for ternary accuracy boost
- Reason: While 99.10% already exceeds 98.9% target, KD can push accuracy further with no latency cost (student architecture unchanged)
- Before: Direct training of ternary_cnn with CE loss only
- After: KD with big-cnn teacher (T=4, alpha=0.7), planned but not yet executed
- Impact: Potential +0.1-0.3pp accuracy improvement with zero inference cost increase
