#!/usr/bin/env python3
"""Exp-002: FashionMNIST architecture sweep — parallel CPU training + latency check."""

import csv
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from multiprocessing import Pool, cpu_count
from pathlib import Path
import threading

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ── Live CSV writer (thread-safe callback) ────────────────────────

LIVE_CSV = RESULTS / "sweep-live.csv"
CSV_FIELDS = ["name", "params", "accuracy", "avg_latency_ms", "p99_latency_ms",
              "train_time_s", "meets_constraint", "status"]
_csv_lock = threading.Lock()
_header_written = False


def _write_live_row(result):
    global _header_written
    with _csv_lock:
        mode = "a" if _header_written else "w"
        with open(LIVE_CSV, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not _header_written:
                writer.writeheader()
                _header_written = True
            writer.writerow(result)


def _on_error(e):
    print(f"  ⚠️ Worker error: {e}")

# ── Model definitions ──────────────────────────────────────────────

class FCNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class CNNNet(nn.Module):
    def __init__(self, channels, fc_layers):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            convs.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.convs = nn.Sequential(*convs)
        spatial = 28 // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for f in fc_layers:
            fc_mods.extend([nn.Linear(prev, f), nn.ReLU()])
            prev = f
        fc_mods.pop()
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DepthwiseCNN(nn.Module):
    def __init__(self, channels, fc_layers):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            if i == 0:
                convs.extend([
                    nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                    nn.ReLU(), nn.MaxPool2d(2),
                ])
            else:
                convs.extend([
                    nn.Conv2d(channels[i], channels[i], 3, padding=1, groups=channels[i]),
                    nn.Conv2d(channels[i], channels[i + 1], 1),
                    nn.ReLU(), nn.MaxPool2d(2),
                ])
        self.convs = nn.Sequential(*convs)
        spatial = 28 // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for f in fc_layers:
            fc_mods.extend([nn.Linear(prev, f), nn.ReLU()])
            prev = f
        fc_mods.pop()
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_model(candidate):
    t = candidate.get("type", "fc")
    if t == "fc":
        return FCNet(candidate["layers"])
    elif t == "cnn":
        return CNNNet(candidate["channels"], candidate["fc"])
    elif t == "depthwise":
        return DepthwiseCNN(candidate["channels"], candidate["fc"])
    else:
        raise ValueError(f"Unknown model type: {t}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── Train & evaluate one candidate ────────────────────────────────

def run_candidate(args):
    candidate, train_cfg, constraint, dataset_name = args
    name = candidate["name"]
    seed = train_cfg["seed"]
    torch.manual_seed(seed)

    try:
        model = build_model(candidate)
        n_params = count_params(model)

        # FashionMNIST normalization (mean=0.2860, std=0.3530)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        ds_cls = getattr(datasets, dataset_name)
        train_ds = ds_cls("./data", train=True, download=True, transform=transform)
        test_ds = ds_cls("./data", train=False, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

        model.train()
        t0 = time.time()
        for epoch in range(train_cfg["epochs"]):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = F.cross_entropy(out, batch_y)
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        accuracy = correct / total

        dummy = torch.randn(1, 1, 28, 28)
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

        meets_constraint = avg_latency_ms <= constraint["max_latency_ms"]

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
        print(f"  ✅ {name}: acc={accuracy:.4f}, latency={avg_latency_ms:.2f}ms, params={n_params}")
        return result

    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return {
            "name": name, "params": 0, "accuracy": 0.0,
            "avg_latency_ms": 999, "p99_latency_ms": 999,
            "train_time_s": 0, "meets_constraint": False,
            "status": f"error: {e}",
        }


# ── Main ──────────────────────────────────────────────────────────

def main():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    candidates = cfg["search"]["candidates"]
    train_cfg = cfg["train"]
    constraint = cfg["constraint"]
    dataset_name = cfg.get("dataset", "FashionMNIST")
    n_workers = min(cfg["search"]["parallel_workers"], cpu_count(), len(candidates))

    print(f"🔍 {dataset_name} Architecture Sweep — {len(candidates)} candidates, {n_workers} workers")
    print(f"   Constraint: ≤{constraint['max_latency_ms']}ms latency (batch_size={constraint['batch_size']})")
    print(f"   Epochs: {train_cfg['epochs']}")
    print()

    args_list = [(c, train_cfg, constraint, dataset_name) for c in candidates]

    if LIVE_CSV.exists():
        LIVE_CSV.unlink()

    with Pool(n_workers) as pool:
        async_results = []
        for args in args_list:
            ar = pool.apply_async(run_candidate, (args,),
                                  callback=_write_live_row,
                                  error_callback=_on_error)
            async_results.append(ar)
        pool.close()
        pool.join()

    results = [ar.get() for ar in async_results]

    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = RESULTS / f"sweep-{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    with open(RESULTS / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    passing = [r for r in results if r["meets_constraint"] and r["status"] == "ok"]
    passing.sort(key=lambda r: r["accuracy"], reverse=True)
    failing = [r for r in results if not r["meets_constraint"] or r["status"] != "ok"]

    print(f"\n{'='*60}")
    print(f"📊 Results: {len(passing)} passed / {len(failing)} failed constraint")
    print(f"{'='*60}")

    if passing:
        print(f"\n🏆 Best: {passing[0]['name']} — accuracy={passing[0]['accuracy']:.4f}, "
              f"latency={passing[0]['avg_latency_ms']:.2f}ms")
        print(f"\nRanking (by accuracy, constraint-passing only):")
        for i, r in enumerate(passing, 1):
            print(f"  {i}. {r['name']:20s}  acc={r['accuracy']:.4f}  "
                  f"lat={r['avg_latency_ms']:.2f}ms  params={r['params']}")

    if failing:
        print(f"\n❌ Eliminated:")
        for r in failing:
            reason = "latency" if r["status"] == "ok" else r["status"]
            print(f"  - {r['name']:20s}  acc={r['accuracy']:.4f}  "
                  f"lat={r['avg_latency_ms']:.2f}ms  reason={reason}")

    summary = {
        "dataset": dataset_name,
        "total": len(results),
        "passing": len(passing),
        "failing": len(failing),
        "best": passing[0] if passing else None,
        "ranking": passing,
        "eliminated": failing,
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 Results saved to {csv_path}")


if __name__ == "__main__":
    main()
