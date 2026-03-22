"""Unified ternary inference wrapper.

Consolidates ternary_engine.py (v1), ternary_v2.py (v2/adder-only),
and ternary_bench.py (v3/zero-overhead benchmark) into a single clean API.

Usage:
    from autolab.inference import TernaryInference

    engine = TernaryInference(version="v3")  # v3 = best (default)
    stats = engine.benchmark(model, n_warmup=200, n_runs=2000)
    pred = engine.predict(model, input_tensor)

Supported versions:
    - "v3" (default): Zero-overhead C benchmark. Entire warmup+benchmark loop
      runs in C with clock_gettime(). Best for accurate latency measurement.
      Backed by ternary_bench.py.
    - "v2": Adder-only engine with bit-packed weights and int16 activations.
      No multiplier in datapath. Backed by ternary_v2.py.
"""

import numpy as np


class TernaryInference:
    """Unified ternary inference API.

    Wraps the underlying v2/v3 engines behind a clean interface.
    Auto-compiles the required C library on first use.

    Args:
        version: Engine version to use. "v3" (default, best) or "v2".
    """

    def __init__(self, version="v3"):
        if version not in ("v2", "v3"):
            raise ValueError(f"Unsupported version '{version}'. Use 'v2' or 'v3'.")
        self._version = version
        # Lazy-load the backend to trigger compilation only when needed
        self._backend = None

    def _ensure_backend(self):
        """Lazy-load the appropriate backend module."""
        if self._backend is not None:
            return
        if self._version == "v3":
            from . import ternary_bench as backend
            self._backend = backend
        else:
            from . import ternary_v2 as backend
            self._backend = backend

    def benchmark(self, model, n_warmup=200, n_runs=2000, is_hybrid=False):
        """Benchmark a PyTorch ternary/hybrid CNN model.

        Extracts ternary weights from the model, runs inference in C,
        and measures latency with zero Python overhead in the hot path.

        Args:
            model: A trained PyTorch model (TernaryCNN, TernaryHybridCNN,
                   CNNBatchNorm, CNNNet, etc.). Weights are ternarized
                   at benchmark time.
            n_warmup: Number of warmup iterations before timing.
            n_runs: Number of timed iterations.
            is_hybrid: If True, FC layers use full-precision weights.

        Returns:
            dict with keys:
                avg_ms: Mean latency in milliseconds.
                median_ms: Median latency in milliseconds.
                p99_ms: 99th percentile latency in milliseconds.
                min_ms: Minimum latency in milliseconds.
                n_runs: Number of timed runs.
                sparsity_info: Per-layer sparsity details.
        """
        self._ensure_backend()

        if self._version == "v3":
            raw = self._backend.benchmark_ternary_model(
                model, is_hybrid=is_hybrid,
                n_warmup=n_warmup, n_runs=n_runs,
            )
        else:  # v2
            raw = self._backend.benchmark_model(
                model, is_hybrid=is_hybrid,
                n_warmup=n_warmup, n_runs=n_runs,
            )

        # Collect sparsity info from model weights
        sparsity_info = self._extract_sparsity(model, is_hybrid)

        return {
            "avg_ms": raw["avg_ms"],
            "median_ms": raw["median_ms"],
            "p99_ms": raw["p99_ms"],
            "min_ms": raw.get("min_ms", raw["avg_ms"]),
            "n_runs": raw.get("n_runs", n_runs),
            "sparsity_info": sparsity_info,
        }

    def predict(self, model, input_tensor):
        """Run single-sample inference and return the predicted class index.

        Uses the v1 TernaryEngine for prediction (supports step-by-step
        layer execution), regardless of the benchmark version selected.

        Args:
            model: A trained PyTorch model.
            input_tensor: Input tensor or numpy array. Shape: (1, H, W),
                          (H, W), or (1, C, H, W). For MNIST: (1, 28, 28).

        Returns:
            int: Predicted class index.
        """
        import torch

        # Normalize input to numpy
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = np.asarray(input_tensor, dtype=np.float32)

        # Remove batch dimension if present: (1, C, H, W) -> (C, H, W)
        if input_np.ndim == 4 and input_np.shape[0] == 1:
            input_np = input_np[0]

        # Determine if hybrid from model class name
        class_name = model.__class__.__name__.lower()
        is_hybrid = "hybrid" in class_name

        # Use TernaryEngine (v1) for step-by-step prediction
        from .ternary_engine import TernaryEngine
        engine = TernaryEngine()
        engine.load_from_pytorch(model, is_hybrid=is_hybrid)
        return engine.predict(input_np)

    @staticmethod
    def _extract_sparsity(model, is_hybrid=False):
        """Extract per-layer sparsity information from model weights."""
        import torch
        state = model.state_dict()
        layers = []

        for key, tensor in state.items():
            if "weight" not in key:
                continue
            if tensor.dim() == 4:  # Conv layer
                w = tensor.detach().cpu().float().numpy()
                abs_w = np.abs(w)
                delta = 0.7 * abs_w.mean()
                n_nonzero = np.sum(abs_w > delta)
                sparsity = 1.0 - n_nonzero / w.size
                alpha = float(abs_w[abs_w > delta].mean()) if n_nonzero > 0 else 0.0
                layers.append({
                    "layer": key,
                    "type": "conv",
                    "shape": list(w.shape),
                    "sparsity": round(float(sparsity), 4),
                    "alpha": round(alpha, 4),
                })
            elif tensor.dim() == 2:  # FC layer
                if is_hybrid or "excite" in key or "squeeze" in key:
                    layers.append({
                        "layer": key,
                        "type": "fc_fp",
                        "shape": list(tensor.shape),
                        "sparsity": 0.0,
                        "alpha": 0.0,
                    })
                else:
                    w = tensor.detach().cpu().float().numpy()
                    abs_w = np.abs(w)
                    delta = 0.7 * abs_w.mean()
                    n_nonzero = np.sum(abs_w > delta)
                    sparsity = 1.0 - n_nonzero / w.size
                    alpha = float(abs_w[abs_w > delta].mean()) if n_nonzero > 0 else 0.0
                    layers.append({
                        "layer": key,
                        "type": "fc_ternary",
                        "shape": list(w.shape),
                        "sparsity": round(float(sparsity), 4),
                        "alpha": round(alpha, 4),
                    })

        return layers

    @staticmethod
    def available_versions():
        """Return list of available engine versions with descriptions."""
        return {
            "v2": "Adder-only engine with bit-packed weights, int16 activations. "
                  "No multiplier in datapath.",
            "v3": "Zero-overhead C benchmark. Entire warmup+benchmark loop in C. "
                  "Best for accurate latency measurement. (default)",
        }

    def __repr__(self):
        return f"TernaryInference(version='{self._version}')"
