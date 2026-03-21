"""Ralph-loop: autonomous iteration engine for neural architecture search."""

import json
import time
import yaml
from pathlib import Path
from datetime import date

from .knowledge import TrackerMD, RegistryMD
from .sweep import SweepRunner
from .safety import check_disk


def load_goal(goal_path):
    """Load goal.yaml and return structured goal dict."""
    with open(goal_path) as f:
        return yaml.safe_load(f)


def check_goal_met(best_result, goal):
    """Check if all goal metrics are met by the best result.

    Returns (met: bool, gaps: dict mapping metric_name -> gap_value).
    """
    metrics = goal.get("metrics", {})
    gaps = {}
    all_met = True

    for metric_name, spec in metrics.items():
        target = spec["target"]
        direction = spec.get("direction", "maximize")

        if best_result is None:
            gaps[metric_name] = target if direction == "maximize" else -target
            all_met = False
            continue

        current = best_result.get(metric_name, 0)

        if direction == "maximize":
            gap = target - current
            if current < target:
                all_met = False
        else:  # minimize
            gap = current - target
            if current > target:
                all_met = False

        gaps[metric_name] = gap

    return all_met, gaps


def select_strategy(gaps, iteration, best_result):
    """Select next strategy based on gap analysis.

    Returns (strategy_name, strategy_description).
    """
    acc_gap = gaps.get("accuracy", 0)

    if iteration == 0 or acc_gap > 0.05:
        return "ArchitectureSearch", "Sweep new model candidates (large accuracy gap or first run)"

    if best_result:
        # Check for overfitting (if we had train accuracy, but we can infer from context)
        acc = best_result.get("accuracy", 0)

        if acc_gap > 0.02:
            return "HyperparameterTuning", "Fine-tune lr, batch_size, optimizer (moderate accuracy gap)"

        if acc_gap > 0:
            return "TrainingExtension", "Increase epochs, add learning rate scheduler"

    lat_gap = gaps.get("avg_latency_ms", 0)
    if lat_gap > 0:
        return "ModelCompression", "Reduce model size to meet latency target"

    return "Refinement", "Minor improvements and ablations"


def find_best_across_experiments(project_dir):
    """Scan all experiment summary.json files and return overall best result."""
    experiments_dir = Path(project_dir) / "experiments"
    best = None
    best_acc = -1

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp-"):
            continue
        summary_path = exp_dir / "results" / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        ranking = summary.get("ranking", [])
        if ranking and ranking[0]["accuracy"] > best_acc:
            best_acc = ranking[0]["accuracy"]
            best = ranking[0]

    return best


def run_ralph(goal_path, project_dir=None):
    """Execute the ralph-loop: autonomous iteration until goal is met.

    Args:
        goal_path: Path to goal.yaml
        project_dir: Project directory (defaults to goal_path parent)
    """
    goal_path = Path(goal_path)
    if project_dir is None:
        project_dir = goal_path.parent
    project_dir = Path(project_dir)

    if not check_disk(str(project_dir)):
        print("DISK FULL — Cannot start ralph-loop")
        return

    goal = load_goal(goal_path)
    max_iterations = goal.get("max_iterations", 10)
    dataset = goal.get("dataset", "MNIST")

    tracker = TrackerMD(project_dir / "experiments" / "TRACKER.md")
    registry = RegistryMD(project_dir / "knowledge" / "REGISTRY.md")

    ralph_log = []
    ralph_log_path = project_dir / "ralph-log.json"

    print("=" * 60)
    print("RALPH-LOOP: Autonomous Neural Architecture Search")
    print(f"Goal: {goal.get('metrics', {})}")
    print(f"Dataset: {dataset}")
    print(f"Max iterations: {max_iterations}")
    print("=" * 60)

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")

        # Step 1: Find current best result across all experiments
        best = find_best_across_experiments(project_dir)

        if best:
            print(f"Current best: {best['name']} acc={best['accuracy']:.4f} "
                  f"lat={best['avg_latency_ms']:.2f}ms")
        else:
            print("No results yet.")

        # Step 2: Check if goal is met
        met, gaps = check_goal_met(best, goal)

        print(f"Gaps: {gaps}")
        print(f"Goal met: {met}")

        if met:
            print("\nGOAL MET! Generating final report...")
            iteration_entry = {
                "iteration": iteration,
                "strategy": "GoalMet",
                "gap": {k: round(v, 4) for k, v in gaps.items()},
                "result": f"Best: {best['name']} acc={best['accuracy']:.4f}" if best else "N/A",
                "best_value": best["accuracy"] if best else 0,
                "reasoning": "All target metrics satisfied. Generating final report.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            ralph_log.append(iteration_entry)

            # Save ralph log
            with open(ralph_log_path, "w") as f:
                json.dump(ralph_log, f, indent=2)

            # Update registry
            registry.append_to_section(
                "Established Facts",
                f"[ralph] Goal met at iteration {iteration}: "
                f"{best['name']} acc={best['accuracy']:.4f}, lat={best['avg_latency_ms']:.2f}ms"
            )

            return {
                "status": "goal_met",
                "iterations": iteration,
                "best": best,
                "log": ralph_log,
            }

        # Step 3: Select strategy
        strategy, description = select_strategy(gaps, iteration, best)
        print(f"Strategy: {strategy} — {description}")

        # Step 4: Execute strategy
        if strategy == "ArchitectureSearch":
            # Load config from project
            config_path = project_dir / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                print("No config.yaml found. Cannot run architecture search.")
                break

            # Register experiment
            exp_num = tracker.next_number()
            exp_name = f"ralph-iter{iteration}-sweep"
            exp_dir = project_dir / "experiments" / f"exp-{exp_num:03d}-{exp_name}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "results").mkdir(exist_ok=True)

            tracker.add_row(
                exp_num, exp_name, "running",
                "pending", date.today().isoformat(),
                f"Ralph iteration {iteration}, strategy={strategy}"
            )

            # Run sweep
            runner = SweepRunner(exp_dir / "results")
            results = runner.run(
                candidates=config["search"]["candidates"],
                train_cfg=config.get("train", {}),
                constraint=config.get("constraint", {}),
                dataset_name=dataset,
                n_workers=config["search"].get("parallel_workers"),
            )

            passing = [r for r in results if r["meets_constraint"] and r["status"] == "ok"]
            passing.sort(key=lambda r: r["accuracy"], reverse=True)

            if passing:
                new_best = passing[0]
                tracker.update_row(
                    exp_num,
                    status="done",
                    metric=f"{new_best['accuracy']*100:.2f}% ({new_best['name']})",
                    notes=f"{len(passing)}/{len(results)} passed"
                )
            else:
                tracker.update_row(exp_num, status="failed", metric="no passing candidates")

        else:
            # For other strategies, log that they would be applied
            print(f"Strategy {strategy} would be applied here (not yet implemented)")

        # Log iteration
        iteration_entry = {
            "iteration": iteration,
            "strategy": strategy,
            "gap": {k: round(v, 4) for k, v in gaps.items()},
            "result": f"Best: {best['name']} acc={best['accuracy']:.4f}" if best else "N/A",
            "best_value": best["accuracy"] if best else 0,
            "reasoning": description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        ralph_log.append(iteration_entry)

        # Save ralph log after each iteration
        with open(ralph_log_path, "w") as f:
            json.dump(ralph_log, f, indent=2)

    print(f"\nRalph-loop completed after {len(ralph_log)} iterations")

    with open(ralph_log_path, "w") as f:
        json.dump(ralph_log, f, indent=2)

    return {
        "status": "max_iterations_reached",
        "iterations": len(ralph_log),
        "best": find_best_across_experiments(project_dir),
        "log": ralph_log,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m autolab.ralph <goal.yaml> [project_dir]")
        sys.exit(1)
    goal = sys.argv[1]
    proj = sys.argv[2] if len(sys.argv) > 2 else None
    run_ralph(goal, proj)
