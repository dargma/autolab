"""Generic sweep engine — parallel architecture search on CPU."""

import csv
import json
import time
import threading
import torch
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
from pathlib import Path

from . import models as model_registry
from . import data as data_factory
from .safety import check_disk

CSV_FIELDS = ["name", "params", "accuracy", "avg_latency_ms", "p99_latency_ms",
              "train_time_s", "meets_constraint", "status"]


def _run_candidate(args):
    """Train and evaluate a single candidate model. Runs in a subprocess."""
    candidate, train_cfg, constraint, dataset_name = args
    name = candidate["name"]
    torch.manual_seed(train_cfg.get("seed", 42))

    try:
        model = model_registry.build_model(candidate)
        n_params = model_registry.count_params(model)

        train_loader, test_loader = data_factory.get_loaders(
            dataset_name,
            batch_size_train=train_cfg.get("batch_size_train", 128),
            batch_size_test=train_cfg.get("batch_size_test", 256),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.001))

        # Train
        model.train()
        t0 = time.time()
        for epoch in range(train_cfg.get("epochs", 5)):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = F.cross_entropy(out, batch_y)
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        # Evaluate accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        accuracy = correct / total

        # Measure latency (single sample, avg over 100 runs)
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
        avg_latency_ms = sum(latencies) / len(latencies)
        p99_latency_ms = sorted(latencies)[98]

        max_lat = constraint.get("max_latency_ms", 100)
        meets_constraint = avg_latency_ms <= max_lat

        result = {
            "name": name,
            "params": n_params,
            "accuracy": round(accuracy, 4),
            "avg_latency_ms": round(avg_latency_ms, 3),
            "p99_latency_ms": round(p99_latency_ms, 3),
            "train_time_s": round(train_time, 1),
            "meets_constraint": meets_constraint,
            "status": "ok",
        }
        print(f"  {name}: acc={accuracy:.4f}, latency={avg_latency_ms:.2f}ms, params={n_params}")
        return result

    except Exception as e:
        print(f"  {name}: ERROR {e}")
        return {
            "name": name, "params": 0, "accuracy": 0.0,
            "avg_latency_ms": 999, "p99_latency_ms": 999,
            "train_time_s": 0, "meets_constraint": False,
            "status": f"error: {e}",
        }


class SweepRunner:
    """Runs a parallel architecture sweep and streams results to CSV."""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._csv_lock = threading.Lock()
        self._header_written = False
        self._live_csv = self.results_dir / "sweep-live.csv"

    def _write_live_row(self, result):
        with self._csv_lock:
            mode = "a" if self._header_written else "w"
            with open(self._live_csv, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(result)

    def _on_error(self, e):
        print(f"  Worker error: {e}")

    def run(self, candidates, train_cfg, constraint, dataset_name, n_workers=None):
        """Run sweep, return list of result dicts sorted by accuracy."""
        if not check_disk(str(self.results_dir)):
            raise RuntimeError("Disk usage too high, aborting sweep")

        if n_workers is None:
            n_workers = min(8, cpu_count(), len(candidates))
        else:
            n_workers = min(n_workers, cpu_count(), len(candidates))

        print(f"Sweep: {len(candidates)} candidates, {n_workers} workers, dataset={dataset_name}")

        args_list = [(c, train_cfg, constraint, dataset_name) for c in candidates]

        if self._live_csv.exists():
            self._live_csv.unlink()
        self._header_written = False

        with Pool(n_workers) as pool:
            async_results = []
            for args in args_list:
                ar = pool.apply_async(
                    _run_candidate, (args,),
                    callback=self._write_live_row,
                    error_callback=self._on_error,
                )
                async_results.append(ar)
            pool.close()
            pool.join()

        results = [ar.get() for ar in async_results]

        # Save timestamped CSV
        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_path = self.results_dir / f"sweep-{ts}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(results)

        # Save JSON
        with open(self.results_dir / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Sort and summarize
        passing = sorted(
            [r for r in results if r["meets_constraint"] and r["status"] == "ok"],
            key=lambda r: r["accuracy"], reverse=True,
        )
        failing = [r for r in results if not r["meets_constraint"] or r["status"] != "ok"]

        summary = {
            "dataset": dataset_name,
            "total": len(results),
            "passing": len(passing),
            "failing": len(failing),
            "best": passing[0] if passing else None,
            "ranking": passing,
            "eliminated": failing,
        }
        with open(self.results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results: {len(passing)} passed / {len(failing)} failed")
        if passing:
            print(f"Best: {passing[0]['name']} acc={passing[0]['accuracy']:.4f} "
                  f"lat={passing[0]['avg_latency_ms']:.2f}ms")

        return results


def run_sweep(project_dir, config):
    """High-level entry: run a sweep from a config dict within a project directory.

    Args:
        project_dir: Path to the project directory
        config: dict with keys: search.candidates, train, constraint, dataset
    Returns:
        list of result dicts
    """
    project_dir = Path(project_dir)
    results_dir = project_dir / "experiments" / "current" / "results"

    runner = SweepRunner(results_dir)
    return runner.run(
        candidates=config["search"]["candidates"],
        train_cfg=config.get("train", {}),
        constraint=config.get("constraint", {}),
        dataset_name=config.get("dataset", "MNIST"),
        n_workers=config["search"].get("parallel_workers"),
    )
