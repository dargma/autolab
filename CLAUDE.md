# Auto-Research System — Claude Code Instructions

## Why This Is Not "Just a Prompt"
- **Persistent state**: REGISTRY.md, TRACKER.md, DECISIONS.md accumulate knowledge across sessions
- **Enforced structure**: Folder/file naming rules survive session resets
- **Reproducibility**: Another researcher can clone this and get the identical workflow

---

## 1. Folder Convention

```
{project}/
├── CLAUDE.md
├── papers/                      # User drops PDFs here
├── knowledge/
│   ├── REGISTRY.md              # Cumulative knowledge (single source of truth)
│   └── DECISIONS.md             # Direction changes / decision history
├── experiments/
│   ├── TRACKER.md               # Experiment status matrix
│   └── exp-{NNN}-{short-name}/
│       ├── config.yaml
│       ├── run.sh
│       ├── results/
│       └── REPORT.md
├── reports/
│   ├── PROGRESS.md              # Cumulative progress report (append-only)
│   └── figures/                 # All figures (SVG/PNG)
└── dashboard.html
```

**Naming rules**:
- Experiments: `exp-001-baseline`, `exp-002-lsr-lambda05`
- Figures: `fig-{exp-number}-{content}.svg`
- Dates: ISO 8601

---

## 2. Core File Rules

### REGISTRY.md — Knowledge Accumulation
- **Never delete entries.** Append or revise only.
- Sections: `## Established Facts`, `## Hypotheses`, `## Rejected Ideas`, `## Open Questions`
- After reading each paper: add a 3-line summary
- After each experiment: update facts/hypotheses

### DECISIONS.md — Direction Change History
- **Must** be updated whenever the user changes direction or plans
- Format: `### YYYY-MM-DD: {title}` + reason + before→after + impact scope
- This file serves as long-term memory

### TRACKER.md — Experiment Matrix
- Table: | # | Name | Status | Key Metric | Date | Notes |
- Status: 🔵planned → 🟡running → 🟢done → 🔴failed

### PROGRESS.md — Cumulative Progress Report
- **CVPR/ECCV quality**: every claim needs a figure or table, baselines always included, one ablation variable at a time, axis labels + legends + captions mandatory
- **Append to existing report** — never create a new file

---

## 3. Disk Safety Guard

**Before every file write or experiment launch:**
```bash
USAGE=$(df -h . | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$USAGE" -ge 95 ]; then
  echo "⛔ DISK 95% — All operations halted. Cleanup required."
  exit 1
fi
```

---

## 4. Session Start Routine

At every new session, **always**:
1. Read `DECISIONS.md` (recent direction changes)
2. Read `REGISTRY.md` (current knowledge state)
3. Read `TRACKER.md` (in-progress experiments)
4. Check disk usage
5. Then begin work

---

## 5. Paper Analysis Protocol

1. Read PDF → extract:
   - **Key contribution** (1 line)
   - **Method** (2 lines max)
   - **Applicability to our work** (1 line)
   - **Limitations / open questions** (1 line)

2. Update REGISTRY.md sections accordingly

3. **Cross-analyze** with existing REGISTRY content:
   - Does the new paper support or refute existing hypotheses?
   - Can its methods combine with what we already know?

---

## 6. Ideation Protocol

⚠️ **Never produce only safe variations.**

Required output:

| # | Name | Novelty | Feasibility | Description |
|---|------|---------|-------------|-------------|
| A | ... | ★★★ | ★★☆ | ... |
| B | ... | ★☆☆ | ★★★ | ... |

Then:
- **Pros/cons comparison** across all candidates
- **Fusion possibilities**: explicitly try "What if we combine A+B?"
- **Cross-domain borrowing**: at least 1 idea from another field (CV, NLP, RL, biology, physics, etc.)
- **Recommendation with rationale**

Checklist:
- [ ] At least one ★★★ novelty idea?
- [ ] At least one cross-domain borrowing?
- [ ] Reviewed fusion of 2+ candidates?
- [ ] Recorded rejections with reasons in REGISTRY?

### Reward Candidate Evaluation (when applicable)
When N reward candidates exist, go beyond comparison:
1. Individual analysis: gradient characteristics, collapse risk, scale sensitivity
2. Pairwise comparison table
3. Elimination with rationale
4. **Fusion design**: combine surviving candidates' strengths into a new reward
5. Validation plan: hypothesis → minimal experiment design

---

## 7. Experiment Lifecycle

### Registration
- Disk 95% check (mandatory)
- Assign next number from TRACKER.md → create `exp-{NNN}-{name}/`
- Write config.yaml: model, hyperparameters, data, hardware, seed

### Execution
- Generate run.sh from config.yaml (fully reproducible)
- GPU: vLLM / HF Trainer standard setup
- CPU test: multiprocessing + constraint checking
- Logs → `results/train.log`, metrics → `results/metrics.csv`

### Analysis
- Parse metrics.csv → extract key numbers
- Update TRACKER.md (status, key metrics)
- Compute delta vs baseline

### Reporting (CVPR Quality)
REPORT.md structure:
```
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
- Colors: colorblind-safe palettes
- 300 DPI, prefer SVG

---

## 8. Dashboard (dashboard.html)

Auto-generate after each experiment completion. Required panels:

1. **Experiment status matrix** — from TRACKER.md (🔵🟡🟢🔴), linked to REPORT.md
2. **Metric trend chart** — X: experiment number, Y: key metric, baseline as horizontal line
3. **Current best result** — best experiment, metric value, improvement over baseline
4. **Resource status** — disk usage (95% warning), experiment count
5. **Research timeline** — direction changes from DECISIONS.md + experiment completions

Implementation: single HTML file, data inlined as JS, Chart.js CDN only, opens via `file://`

---

## 9. Multi-CPU Architecture Search

For fast model search without GPU:

```yaml
task: mnist_arch_search
constraint:
  fps: 60
  device: cpu
  batch_size: 1
search:
  parallel_workers: 8
  candidates: [...]
  metric: accuracy
  budget: 300s
```

Flow: generate candidates → parallel train via multiprocessing.Pool → discard constraint failures → sort by metric → save `results/sweep-{timestamp}.csv` → update REPORT.md + dashboard

---

## 10. Relationship with /ralph-loop

**Independent.** ralph-loop = code iteration (implementation level). This system = experiment management (research level). run.sh may invoke ralph-loop internally, but the system never depends on it.
