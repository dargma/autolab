"""Ralph-loop: autonomous iteration engine for neural architecture search."""

import json
import time
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import date

from .knowledge import TrackerMD, RegistryMD
from .sweep import SweepRunner
from .safety import check_disk
from . import models as model_registry
from . import data as data_factory


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


def _best_model_config(best_result):
    """Reconstruct a candidate config dict from the best result's name.

    Parses model names like 'cnn_bn-16-32-64-10' into a config dict.
    Falls back to a reasonable default if parsing fails.
    """
    name = best_result.get("name", "")
    # Try to infer model type and architecture from the name
    # Common patterns: cnn_bn-16-32-fc128-10, residual_cnn-16-32-fc64-10
    for model_type in ["residual_cnn", "squeeze_excite_cnn", "cnn_bn", "cnn",
                        "ternary_hybrid_cnn", "ternary_cnn", "depthwise", "fc"]:
        if name.startswith(model_type):
            return {"type": model_type, "name": name}
    # Default fallback
    return {"type": "cnn_bn", "name": name}


def _find_best_checkpoint(project_dir):
    """Find the best model checkpoint (.pt/.pth) across experiments."""
    experiments_dir = Path(project_dir) / "experiments"
    best_ckpt = None
    best_acc = -1

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp-"):
            continue
        # Check for checkpoints
        ckpt_dir = exp_dir / "results" / "checkpoints"
        if ckpt_dir.exists():
            for ckpt in sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth")):
                # Try to extract accuracy from filename or summary
                if best_ckpt is None:
                    best_ckpt = ckpt
        # Also check for summary to get accuracy
        summary_path = exp_dir / "results" / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            ranking = summary.get("ranking", [])
            if ranking and ranking[0].get("accuracy", 0) > best_acc:
                best_acc = ranking[0]["accuracy"]
                # Look for checkpoint in this experiment
                ckpt_dir = exp_dir / "results" / "checkpoints"
                if ckpt_dir.exists():
                    ckpts = sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth"))
                    if ckpts:
                        best_ckpt = ckpts[-1]  # Latest checkpoint

    return best_ckpt, best_acc


def _train_single_model(model, train_cfg, dataset_name):
    """Train a single model and return (accuracy, model, train_time)."""
    torch.manual_seed(train_cfg.get("seed", 42))

    # Data
    use_aug = train_cfg.get("augmentation", False)
    batch_size = train_cfg.get("batch_size_train", 128)
    if use_aug and dataset_name in data_factory.DATASETS:
        from torchvision import transforms as T
        ds_cls, mean, std, ch, sz, nc = data_factory.DATASETS[dataset_name]
        aug_transform = T.Compose([
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(), T.Normalize(mean, std),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
        train_ds = ds_cls("./data", train=True, download=True, transform=aug_transform)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        train_loader, _ = data_factory.get_loaders(
            dataset_name, batch_size_train=batch_size)
    _, test_loader = data_factory.get_loaders(dataset_name, batch_size_test=256)

    # Optimizer
    opt_name = train_cfg.get("optimizer", "adam").lower()
    lr = train_cfg.get("lr", 0.001)
    wd = train_cfg.get("weight_decay", 0)
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    epochs = train_cfg.get("epochs", 5)
    sched_name = train_cfg.get("scheduler", "none").lower()
    scheduler = None
    if sched_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(optimizer, max_lr=lr * 10,
                               steps_per_epoch=len(train_loader), epochs=epochs)

    label_smoothing = train_cfg.get("label_smoothing", 0.0)

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.cross_entropy(out, batch_y, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            if sched_name == "onecycle" and scheduler is not None:
                scheduler.step()
        if sched_name == "cosine" and scheduler is not None:
            scheduler.step()
    train_time = time.time() - t0

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = correct / total

    return accuracy, model, train_time


def _measure_latency(model, dataset_name, n_warmup=10, n_runs=100):
    """Measure inference latency for a model."""
    ch, sz, _ = data_factory.get_info(dataset_name)
    dummy = torch.randn(1, ch, sz, sz)
    model.eval()
    for _ in range(n_warmup):
        model(dummy)
    latencies = []
    for _ in range(n_runs):
        t1 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        t2 = time.perf_counter()
        latencies.append((t2 - t1) * 1000)
    latencies.sort()
    return {
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "p99_latency_ms": round(latencies[int(0.99 * len(latencies))], 3),
    }


def run_hyperparameter_tuning(project_dir, best_result, tracker, dataset, goal):
    """Take the best model config, perturb lr/batch_size/optimizer, run mini-sweep.

    Generates variants by:
    - lr: 0.5x and 2x of baseline (default 0.001)
    - batch_size: 64 and 256
    - optimizer: Adam, AdamW, SGD+momentum
    """
    project_dir = Path(project_dir)
    if not check_disk(str(project_dir)):
        print("DISK FULL — Cannot run hyperparameter tuning")
        return None

    base_config = _best_model_config(best_result)
    config_path = project_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            project_config = yaml.safe_load(f)
    else:
        project_config = {}

    # Build candidate list: all combos of hp perturbations
    base_lr = project_config.get("train", {}).get("lr", 0.001)
    base_epochs = project_config.get("train", {}).get("epochs", 5)
    base_candidates = project_config.get("search", {}).get("candidates", [])

    # Find the candidate config matching the best result
    best_candidate = None
    for c in base_candidates:
        if c.get("name") == best_result.get("name"):
            best_candidate = c
            break
    if best_candidate is None:
        # Use the inferred config
        best_candidate = base_config

    hp_variants = []
    variant_id = 0
    for lr_mult, lr_label in [(0.5, "lr05x"), (2.0, "lr2x"), (1.0, "lr1x")]:
        for bs in [64, 256]:
            for opt in ["adam", "adamw", "sgd"]:
                variant_id += 1
                c = copy.deepcopy(best_candidate)
                c["name"] = f"hp-{lr_label}-bs{bs}-{opt}"
                hp_variants.append({
                    "candidate": c,
                    "train_cfg": {
                        "lr": base_lr * lr_mult,
                        "batch_size_train": bs,
                        "optimizer": opt,
                        "epochs": base_epochs,
                        "weight_decay": 1e-4 if opt in ("adamw", "sgd") else 0,
                        "seed": 42,
                    },
                })

    # Register experiment
    exp_num = tracker.next_number()
    exp_name = "ralph-hp-tuning"
    exp_dir = project_dir / "experiments" / f"exp-{exp_num:03d}-{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    tracker.add_row(
        exp_num, exp_name, "running",
        "pending", date.today().isoformat(),
        "Hyperparameter tuning on best model"
    )

    # Write config
    config_out = {
        "strategy": "HyperparameterTuning",
        "base_model": best_result.get("name", "unknown"),
        "variants": [{"name": v["candidate"]["name"], **v["train_cfg"]}
                     for v in hp_variants],
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_out, f, default_flow_style=False)

    # Build sweep candidates (reuse the same model architecture, vary training)
    constraint = project_config.get("constraint", {})
    sweep_candidates = [v["candidate"] for v in hp_variants]
    # Use the first variant's train_cfg as base, but we actually need per-candidate cfgs
    # SweepRunner uses a single train_cfg, so we create candidates with embedded train params
    # by running them sequentially
    all_results = []
    print(f"HP Tuning: {len(hp_variants)} variants")
    for v in hp_variants:
        try:
            model = model_registry.build_model(v["candidate"])
            acc, model, train_time = _train_single_model(
                model, v["train_cfg"], dataset)
            lat_info = _measure_latency(model, dataset)
            max_lat = constraint.get("max_latency_ms", 100)
            result = {
                "name": v["candidate"]["name"],
                "params": model_registry.count_params(model),
                "accuracy": round(acc, 4),
                "avg_latency_ms": lat_info["avg_latency_ms"],
                "p99_latency_ms": lat_info["p99_latency_ms"],
                "train_time_s": round(train_time, 1),
                "meets_constraint": lat_info["avg_latency_ms"] <= max_lat,
                "status": "ok",
            }
            print(f"  {result['name']}: acc={acc:.4f}")
            all_results.append(result)
        except Exception as e:
            print(f"  {v['candidate']['name']}: ERROR {e}")
            all_results.append({
                "name": v["candidate"]["name"], "params": 0, "accuracy": 0.0,
                "avg_latency_ms": 999, "p99_latency_ms": 999,
                "train_time_s": 0, "meets_constraint": False, "status": f"error: {e}",
            })

    # Save results
    passing = sorted(
        [r for r in all_results if r["meets_constraint"] and r["status"] == "ok"],
        key=lambda r: r["accuracy"], reverse=True,
    )
    summary = {
        "dataset": dataset, "total": len(all_results),
        "passing": len(passing), "ranking": passing,
        "best": passing[0] if passing else None,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "exp_num": exp_num,
        "best": passing[0] if passing else None,
        "n_passing": len(passing),
        "n_total": len(all_results),
    }


def run_training_extension(project_dir, best_result, tracker, dataset, goal):
    """Load the best checkpoint and continue training with reduced LR."""
    project_dir = Path(project_dir)
    if not check_disk(str(project_dir)):
        print("DISK FULL — Cannot run training extension")
        return None

    config_path = project_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            project_config = yaml.safe_load(f)
    else:
        project_config = {}

    # Find best checkpoint
    best_ckpt, ckpt_acc = _find_best_checkpoint(project_dir)

    # Determine model config
    base_config = _best_model_config(best_result)
    base_candidates = project_config.get("search", {}).get("candidates", [])
    best_candidate = None
    for c in base_candidates:
        if c.get("name") == best_result.get("name"):
            best_candidate = c
            break
    if best_candidate is None:
        best_candidate = base_config

    # Register experiment
    exp_num = tracker.next_number()
    exp_name = "ralph-train-extend"
    exp_dir = project_dir / "experiments" / f"exp-{exp_num:03d}-{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)

    base_lr = project_config.get("train", {}).get("lr", 0.001)
    base_epochs = project_config.get("train", {}).get("epochs", 5)
    extra_epochs = max(base_epochs, 10)  # At least double the original epochs

    tracker.add_row(
        exp_num, exp_name, "running",
        "pending", date.today().isoformat(),
        f"Extend training {extra_epochs} epochs, reduced LR"
    )

    # Write config
    train_cfg = {
        "lr": base_lr * 0.1,  # 10x reduced LR
        "epochs": extra_epochs,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "batch_size_train": project_config.get("train", {}).get("batch_size_train", 128),
        "augmentation": True,
        "label_smoothing": 0.05,
        "seed": 42,
    }
    config_out = {
        "strategy": "TrainingExtension",
        "base_model": best_result.get("name", "unknown"),
        "checkpoint": str(best_ckpt) if best_ckpt else "none",
        "train": train_cfg,
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_out, f, default_flow_style=False)

    try:
        # Build model
        model = model_registry.build_model(best_candidate)

        # Load checkpoint if available
        if best_ckpt and best_ckpt.exists():
            print(f"Loading checkpoint: {best_ckpt}")
            state_dict = torch.load(best_ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

        # Continue training
        accuracy, model, train_time = _train_single_model(model, train_cfg, dataset)

        # Save checkpoint
        ckpt_path = results_dir / "checkpoints" / "extended_best.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Measure latency
        lat_info = _measure_latency(model, dataset)

        # Save summary
        summary = {
            "dataset": dataset,
            "accuracy": round(accuracy, 4),
            "avg_latency_ms": lat_info["avg_latency_ms"],
            "train_time_s": round(train_time, 1),
            "extra_epochs": extra_epochs,
            "ranking": [{
                "name": best_result.get("name", "extended"),
                "accuracy": round(accuracy, 4),
                "avg_latency_ms": lat_info["avg_latency_ms"],
            }],
        }
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Training extension: acc={accuracy:.4f} (was {best_result.get('accuracy', 0):.4f})")

        return {
            "exp_num": exp_num,
            "accuracy": accuracy,
            "avg_latency_ms": lat_info["avg_latency_ms"],
            "extra_epochs": extra_epochs,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training extension failed: {e}")
        return {"exp_num": exp_num}


def run_model_compression(project_dir, best_result, tracker, dataset, goal):
    """Apply ternary quantization to the best model. Train with STE.

    Optionally runs knowledge distillation using autolab.distill.
    """
    project_dir = Path(project_dir)
    if not check_disk(str(project_dir)):
        print("DISK FULL — Cannot run model compression")
        return None

    config_path = project_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            project_config = yaml.safe_load(f)
    else:
        project_config = {}

    # Find the best model's candidate config
    base_candidates = project_config.get("search", {}).get("candidates", [])
    best_candidate = None
    for c in base_candidates:
        if c.get("name") == best_result.get("name"):
            best_candidate = c
            break
    if best_candidate is None:
        best_candidate = _best_model_config(best_result)

    # Register experiment
    exp_num = tracker.next_number()
    exp_name = "ralph-compression"
    exp_dir = project_dir / "experiments" / f"exp-{exp_num:03d}-{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)

    base_lr = project_config.get("train", {}).get("lr", 0.001)
    base_epochs = project_config.get("train", {}).get("epochs", 5)

    tracker.add_row(
        exp_num, exp_name, "running",
        "pending", date.today().isoformat(),
        "Ternary quantization + optional KD"
    )

    # Build ternary version of the architecture
    # Map standard types to ternary equivalents
    ternary_candidate = copy.deepcopy(best_candidate)
    orig_type = ternary_candidate.get("type", "cnn_bn")
    if orig_type in ("cnn", "cnn_bn", "residual_cnn", "squeeze_excite_cnn"):
        ternary_candidate["type"] = "ternary_cnn"
    ternary_candidate["name"] = f"ternary-{best_result.get('name', 'model')}"

    # Also try hybrid variant
    hybrid_candidate = copy.deepcopy(best_candidate)
    if orig_type in ("cnn", "cnn_bn", "residual_cnn", "squeeze_excite_cnn"):
        hybrid_candidate["type"] = "ternary_hybrid_cnn"
    hybrid_candidate["name"] = f"hybrid-{best_result.get('name', 'model')}"

    config_out = {
        "strategy": "ModelCompression",
        "base_model": best_result.get("name", "unknown"),
        "ternary_model": ternary_candidate["name"],
        "hybrid_model": hybrid_candidate["name"],
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_out, f, default_flow_style=False)

    train_cfg = {
        "lr": base_lr,
        "epochs": max(base_epochs, 10),  # More epochs for ternary convergence
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "batch_size_train": 128,
        "augmentation": True,
        "label_smoothing": 0.05,
        "seed": 42,
    }

    all_results = []

    # Train ternary and hybrid variants
    for candidate in [ternary_candidate, hybrid_candidate]:
        try:
            model = model_registry.build_model(candidate)
            acc, model, train_time = _train_single_model(model, train_cfg, dataset)
            lat_info = _measure_latency(model, dataset)

            # Save checkpoint
            ckpt_path = results_dir / "checkpoints" / f"{candidate['name']}.pt"
            torch.save(model.state_dict(), ckpt_path)

            result = {
                "name": candidate["name"],
                "params": model_registry.count_params(model),
                "accuracy": round(acc, 4),
                "avg_latency_ms": lat_info["avg_latency_ms"],
                "p99_latency_ms": lat_info["p99_latency_ms"],
                "train_time_s": round(train_time, 1),
                "meets_constraint": True,
                "status": "ok",
            }
            print(f"  {candidate['name']}: acc={acc:.4f}, lat={lat_info['avg_latency_ms']:.2f}ms")
            all_results.append(result)

            # Try knowledge distillation if we have a teacher
            best_ckpt, _ = _find_best_checkpoint(project_dir)
            if best_ckpt and best_ckpt.exists():
                try:
                    from .distill import distill_train
                    # Build teacher from original architecture
                    teacher = model_registry.build_model(best_candidate)
                    teacher_state = torch.load(best_ckpt, map_location="cpu",
                                               weights_only=True)
                    teacher.load_state_dict(teacher_state)

                    # Build fresh student
                    student = model_registry.build_model(candidate)
                    kd_acc, student = distill_train(
                        teacher, student, dataset_name=dataset,
                        epochs=max(base_epochs, 10), lr=base_lr,
                    )

                    kd_lat = _measure_latency(student, dataset)
                    kd_name = f"kd-{candidate['name']}"
                    kd_result = {
                        "name": kd_name,
                        "params": model_registry.count_params(student),
                        "accuracy": round(kd_acc, 4),
                        "avg_latency_ms": kd_lat["avg_latency_ms"],
                        "p99_latency_ms": kd_lat["p99_latency_ms"],
                        "train_time_s": round(train_time, 1),
                        "meets_constraint": True,
                        "status": "ok",
                    }
                    print(f"  {kd_name}: acc={kd_acc:.4f} (KD)")
                    all_results.append(kd_result)

                    # Save KD checkpoint
                    ckpt_path = results_dir / "checkpoints" / f"{kd_name}.pt"
                    torch.save(student.state_dict(), ckpt_path)
                except Exception as e:
                    print(f"  KD failed for {candidate['name']}: {e}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {candidate['name']}: ERROR {e}")
            all_results.append({
                "name": candidate["name"], "params": 0, "accuracy": 0.0,
                "avg_latency_ms": 999, "p99_latency_ms": 999,
                "train_time_s": 0, "meets_constraint": False, "status": f"error: {e}",
            })

    # Save summary
    passing = sorted(
        [r for r in all_results if r["status"] == "ok"],
        key=lambda r: r["accuracy"], reverse=True,
    )
    summary = {
        "dataset": dataset, "total": len(all_results),
        "passing": len(passing), "ranking": passing,
        "best": passing[0] if passing else None,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best = passing[0] if passing else None
    return {
        "exp_num": exp_num,
        "accuracy": best["accuracy"] if best else None,
        "avg_latency_ms": best["avg_latency_ms"] if best else None,
    }


def run_refinement(project_dir, best_result, tracker, dataset, goal):
    """Combine augmentation + label smoothing + cosine LR on the best architecture."""
    project_dir = Path(project_dir)
    if not check_disk(str(project_dir)):
        print("DISK FULL — Cannot run refinement")
        return None

    config_path = project_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            project_config = yaml.safe_load(f)
    else:
        project_config = {}

    base_candidates = project_config.get("search", {}).get("candidates", [])
    best_candidate = None
    for c in base_candidates:
        if c.get("name") == best_result.get("name"):
            best_candidate = c
            break
    if best_candidate is None:
        best_candidate = _best_model_config(best_result)

    # Register experiment
    exp_num = tracker.next_number()
    exp_name = "ralph-refinement"
    exp_dir = project_dir / "experiments" / f"exp-{exp_num:03d}-{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    base_lr = project_config.get("train", {}).get("lr", 0.001)
    base_epochs = project_config.get("train", {}).get("epochs", 5)

    tracker.add_row(
        exp_num, exp_name, "running",
        "pending", date.today().isoformat(),
        "Refinement: aug + label smoothing + cosine LR"
    )

    # Generate refinement variants
    refinement_configs = [
        {
            "name": "refine-aug-ls01-cosine",
            "train": {
                "lr": base_lr, "epochs": max(base_epochs * 2, 10),
                "optimizer": "adamw", "weight_decay": 1e-4,
                "scheduler": "cosine", "augmentation": True,
                "label_smoothing": 0.1, "batch_size_train": 128, "seed": 42,
            },
        },
        {
            "name": "refine-aug-ls005-cosine",
            "train": {
                "lr": base_lr, "epochs": max(base_epochs * 2, 10),
                "optimizer": "adamw", "weight_decay": 1e-4,
                "scheduler": "cosine", "augmentation": True,
                "label_smoothing": 0.05, "batch_size_train": 128, "seed": 42,
            },
        },
        {
            "name": "refine-aug-ls01-onecycle",
            "train": {
                "lr": base_lr, "epochs": max(base_epochs * 2, 10),
                "optimizer": "adamw", "weight_decay": 1e-4,
                "scheduler": "onecycle", "augmentation": True,
                "label_smoothing": 0.1, "batch_size_train": 128, "seed": 42,
            },
        },
        {
            "name": "refine-heavy-aug-cosine",
            "train": {
                "lr": base_lr * 0.5, "epochs": max(base_epochs * 3, 15),
                "optimizer": "adamw", "weight_decay": 5e-4,
                "scheduler": "cosine", "augmentation": True,
                "label_smoothing": 0.1, "batch_size_train": 64, "seed": 42,
            },
        },
    ]

    config_out = {
        "strategy": "Refinement",
        "base_model": best_result.get("name", "unknown"),
        "variants": refinement_configs,
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_out, f, default_flow_style=False)

    constraint = project_config.get("constraint", {})
    all_results = []

    for rc in refinement_configs:
        try:
            model = model_registry.build_model(best_candidate)
            acc, model, train_time = _train_single_model(model, rc["train"], dataset)
            lat_info = _measure_latency(model, dataset)
            max_lat = constraint.get("max_latency_ms", 100)

            result = {
                "name": rc["name"],
                "params": model_registry.count_params(model),
                "accuracy": round(acc, 4),
                "avg_latency_ms": lat_info["avg_latency_ms"],
                "p99_latency_ms": lat_info["p99_latency_ms"],
                "train_time_s": round(train_time, 1),
                "meets_constraint": lat_info["avg_latency_ms"] <= max_lat,
                "status": "ok",
            }
            print(f"  {rc['name']}: acc={acc:.4f}")
            all_results.append(result)
        except Exception as e:
            print(f"  {rc['name']}: ERROR {e}")
            all_results.append({
                "name": rc["name"], "params": 0, "accuracy": 0.0,
                "avg_latency_ms": 999, "p99_latency_ms": 999,
                "train_time_s": 0, "meets_constraint": False, "status": f"error: {e}",
            })

    # Save results
    passing = sorted(
        [r for r in all_results if r["meets_constraint"] and r["status"] == "ok"],
        key=lambda r: r["accuracy"], reverse=True,
    )
    summary = {
        "dataset": dataset, "total": len(all_results),
        "passing": len(passing), "ranking": passing,
        "best": passing[0] if passing else None,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "exp_num": exp_num,
        "best": passing[0] if passing else None,
        "n_passing": len(passing),
        "n_total": len(all_results),
    }


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

        elif strategy == "HyperparameterTuning":
            result = run_hyperparameter_tuning(
                project_dir, best, tracker, dataset, goal)
            if result and result.get("best"):
                new_best = result["best"]
                tracker.update_row(
                    result["exp_num"], status="done",
                    metric=f"{new_best['accuracy']*100:.2f}% ({new_best['name']})",
                    notes=f"hp-tuning, {result['n_passing']}/{result['n_total']} passed"
                )
            elif result:
                tracker.update_row(result["exp_num"], status="failed",
                                   metric="no improvement")

        elif strategy == "TrainingExtension":
            result = run_training_extension(
                project_dir, best, tracker, dataset, goal)
            if result and result.get("accuracy"):
                tracker.update_row(
                    result["exp_num"], status="done",
                    metric=f"{result['accuracy']*100:.2f}%",
                    notes=f"extended {result['extra_epochs']} epochs"
                )
            elif result:
                tracker.update_row(result["exp_num"], status="failed",
                                   metric="extension failed")

        elif strategy == "ModelCompression":
            result = run_model_compression(
                project_dir, best, tracker, dataset, goal)
            if result and result.get("accuracy"):
                tracker.update_row(
                    result["exp_num"], status="done",
                    metric=f"{result['accuracy']*100:.2f}%",
                    notes=f"ternary, lat={result.get('avg_latency_ms', 'N/A')}ms"
                )
            elif result:
                tracker.update_row(result["exp_num"], status="failed",
                                   metric="compression failed")

        elif strategy == "Refinement":
            result = run_refinement(
                project_dir, best, tracker, dataset, goal)
            if result and result.get("best"):
                new_best = result["best"]
                tracker.update_row(
                    result["exp_num"], status="done",
                    metric=f"{new_best['accuracy']*100:.2f}% ({new_best['name']})",
                    notes=f"refinement, {result['n_passing']}/{result['n_total']} passed"
                )
            elif result:
                tracker.update_row(result["exp_num"], status="failed",
                                   metric="no improvement")

        else:
            print(f"Strategy {strategy} not recognized, skipping")

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
