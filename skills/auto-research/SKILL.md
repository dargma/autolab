---
name: auto-research
description: Full research workflow — survey, ideation, experimentation, synthesis
user_invocable: true
---

# Auto-Research Skill

Manages the four-phase research workflow: survey papers, generate ideas, run experiments, synthesize results.

## Commands

### `/research survey [project_dir]`
Read papers and build knowledge.
1. Scan `papers/` directory for unprocessed PDFs
2. For each paper, extract: key contribution, method, applicability, limitations
3. Update `knowledge/REGISTRY.md` with findings
4. Cross-analyze: does the paper support/refute existing hypotheses?
5. Update `## Open Questions` with new research directions

### `/research ideate [project_dir]`
Generate experiment ideas with novelty ratings.
1. Read current REGISTRY.md and TRACKER.md
2. Produce idea table:

| # | Name | Novelty | Feasibility | Description |
|---|------|---------|-------------|-------------|

3. Requirements:
   - At least 2 safe improvements + 2 bold fusion ideas + 1 cross-domain transplant
   - Pros/cons comparison across all candidates
   - Fusion possibilities: "What if we combine A+B?"
   - Cross-domain borrowing from another field
4. Record rejected ideas with rationale in REGISTRY.md `## Rejected Ideas`
5. Output: recommended next experiment with justification

### `/research run [project_dir]`
Execute the next planned experiment.
1. Check TRACKER.md for next 🔵planned experiment
2. If none planned, run `/research ideate` first
3. Register experiment: assign number, create `exp-{NNN}-{name}/`
4. Disk safety check
5. Write config.yaml and run.sh
6. Execute sweep or single training run
7. Parse results, update TRACKER.md (🟡running → 🟢done or 🔴failed)
8. Generate REPORT.md for the experiment
9. Refresh dashboard

### `/research synthesize [project_dir]`
Update cumulative progress with latest results.
1. Review all experiments in TRACKER.md
2. Append new results to `reports/PROGRESS.md`
3. Generate comparison figures (best vs baseline)
4. Update REGISTRY.md: promote hypotheses to facts or rejections
5. Propose next steps in `## Open Questions`
6. Refresh dashboard.html

## Usage

```bash
# From Claude Code
/research survey projects/cifar-fast
/research ideate projects/cifar-fast
/research run projects/cifar-fast
/research synthesize projects/cifar-fast
```

## Phase Flow

```
Survey → Ideate → Run → Synthesize → (loop)
   ↑                                    |
   └────────────────────────────────────┘
```

Each phase reads from and writes to the persistent knowledge files (REGISTRY.md, TRACKER.md, DECISIONS.md, PROGRESS.md), maintaining continuity across sessions.
