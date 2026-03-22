"""CLI entry point for autolab: python -m autolab <subcommand>."""

import argparse
import sys
from pathlib import Path


def cmd_new(args):
    """Scaffold a new project."""
    from .scaffold import create_project
    create_project(args.project_name, base_dir=args.base_dir)


def cmd_sweep(args):
    """Run sweep from config.yaml."""
    import yaml
    from .safety import check_disk
    from .sweep import run_sweep

    project_dir = Path(args.project_dir)
    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    if not check_disk(str(project_dir)):
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_sweep(project_dir, config)


def cmd_ralph(args):
    """Run ralph-loop from goal.yaml."""
    from .ralph import run_ralph

    project_dir = Path(args.project_dir)
    goal_path = project_dir / "goal.yaml"
    if not goal_path.exists():
        print(f"Error: {goal_path} not found")
        sys.exit(1)

    run_ralph(goal_path, project_dir)


def cmd_bench(args):
    """Benchmark a model checkpoint."""
    import time
    import torch
    from .safety import check_disk
    from . import data as data_factory

    project_dir = Path(args.project_dir)
    checkpoint = Path(args.model_checkpoint)

    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}")
        sys.exit(1)

    if not check_disk(str(project_dir)):
        sys.exit(1)

    # Load goal for dataset info
    goal_path = project_dir / "goal.yaml"
    dataset_name = "MNIST"
    if goal_path.exists():
        import yaml
        with open(goal_path) as f:
            goal = yaml.safe_load(f)
        dataset_name = goal.get("dataset", "MNIST")

    # Load model
    model = torch.jit.load(str(checkpoint)) if str(checkpoint).endswith(".pt") else torch.load(str(checkpoint))
    model.eval()

    # Accuracy
    _, test_loader = data_factory.get_loaders(dataset_name)
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = correct / total

    # Latency
    ch, sz, _ = data_factory.get_info(dataset_name)
    dummy = torch.randn(1, ch, sz, sz)
    for _ in range(10):
        model(dummy)
    latencies = []
    for _ in range(100):
        t1 = time.perf_counter()
        model(dummy)
        t2 = time.perf_counter()
        latencies.append((t2 - t1) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    p99_lat = sorted(latencies)[98]

    print(f"Model:    {checkpoint.name}")
    print(f"Dataset:  {dataset_name}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Latency:  {avg_lat:.3f}ms avg, {p99_lat:.3f}ms p99")


def cmd_dashboard(args):
    """Generate dashboard.html."""
    from .dashboard import generate_dashboard
    generate_dashboard(args.project_dir)


def cmd_report(args):
    """Generate figures and update PROGRESS.md."""
    import json
    import yaml
    from pathlib import Path
    from .figures import plot_sweep_comparison, plot_pareto, load_results
    from .safety import check_disk

    project_dir = Path(args.project_dir)
    if not check_disk(str(project_dir)):
        sys.exit(1)

    figures_dir = project_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load goal for targets
    goal_path = project_dir / "goal.yaml"
    target_acc = None
    if goal_path.exists():
        with open(goal_path) as f:
            goal = yaml.safe_load(f)
        metrics = goal.get("metrics", {})
        if "accuracy" in metrics:
            target_acc = metrics["accuracy"]["target"]

    # Find all experiment results
    experiments_dir = project_dir / "experiments"
    all_results = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp-"):
            continue
        summary_path = exp_dir / "results" / "summary.json"
        if summary_path.exists():
            results = load_results(summary_path)
            all_results.extend(results)

    if not all_results:
        print("No experiment results found.")
        return

    # Generate figures
    plot_sweep_comparison(
        all_results, target_accuracy=target_acc,
        output_path=str(figures_dir / "fig-sweep-comparison.svg"),
        title=f"{project_dir.name}: Accuracy Comparison",
    )
    plot_pareto(
        all_results,
        output_path=str(figures_dir / "fig-pareto.svg"),
        title=f"{project_dir.name}: Pareto Frontier",
    )

    # Append to PROGRESS.md
    progress_path = project_dir / "reports" / "PROGRESS.md"
    best = max(all_results, key=lambda r: r["accuracy"])

    from datetime import date
    entry = f"""
## Update {date.today().isoformat()}

**Models evaluated:** {len(all_results)}
**Best result:** {best['name']} — {best['accuracy']*100:.2f}% accuracy, {best['avg_latency_ms']:.2f}ms latency

![Accuracy Comparison](figures/fig-sweep-comparison.svg)
![Pareto Frontier](figures/fig-pareto.svg)
"""
    existing = progress_path.read_text() if progress_path.exists() else ""
    progress_path.write_text(existing.rstrip() + "\n" + entry)
    print(f"Updated: {progress_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="autolab",
        description="Autonomous AI lab for neural architecture search",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # new
    p_new = sub.add_parser("new", help="Scaffold a new project")
    p_new.add_argument("project_name", help="Name for the new project")
    p_new.add_argument("--base-dir", default="projects", help="Base directory (default: projects)")
    p_new.set_defaults(func=cmd_new)

    # sweep
    p_sweep = sub.add_parser("sweep", help="Run sweep from config.yaml")
    p_sweep.add_argument("project_dir", help="Path to the project directory")
    p_sweep.set_defaults(func=cmd_sweep)

    # ralph
    p_ralph = sub.add_parser("ralph", help="Run ralph-loop from goal.yaml")
    p_ralph.add_argument("project_dir", help="Path to the project directory")
    p_ralph.set_defaults(func=cmd_ralph)

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark a model checkpoint")
    p_bench.add_argument("project_dir", help="Path to the project directory")
    p_bench.add_argument("model_checkpoint", help="Path to model checkpoint file")
    p_bench.set_defaults(func=cmd_bench)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Generate dashboard.html")
    p_dash.add_argument("project_dir", help="Path to the project directory")
    p_dash.set_defaults(func=cmd_dashboard)

    # report
    p_report = sub.add_parser("report", help="Generate figures and update PROGRESS.md")
    p_report.add_argument("project_dir", help="Path to the project directory")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
