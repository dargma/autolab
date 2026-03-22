"""Zero-overhead ternary CNN benchmark — single C call for all N inferences.

The entire warmup + benchmark loop runs in C with clock_gettime().
No Python/ctypes overhead in the hot path.
"""

import ctypes
import numpy as np
import time
from pathlib import Path

_LIB = None


def _get_lib():
    global _LIB
    if _LIB is None:
        lib_path = Path(__file__).parent / "csrc" / "libternary_bench.so"
        if not lib_path.exists():
            import subprocess
            src = lib_path.parent / "ternary_bench.c"
            subprocess.run([
                "gcc", "-O3", "-march=native", "-ffast-math",
                "-shared", "-fPIC", "-o", str(lib_path), str(src), "-lm"
            ], check=True)
        _LIB = ctypes.CDLL(str(lib_path))

        _LIB.benchmark_forward.argtypes = [
            # Conv1: weights, alpha, bias, bn params, out_ch
            ctypes.POINTER(ctypes.c_int8), ctypes.c_float, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            # Conv2
            ctypes.POINTER(ctypes.c_int8), ctypes.c_float, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            # FC1: ternary_weights, alpha, bias, fp_weights, out
            ctypes.POINTER(ctypes.c_int8), ctypes.c_float, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            # FC2
            ctypes.POINTER(ctypes.c_int8), ctypes.c_float, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            # Input, config, output
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        _LIB.benchmark_forward.restype = None
    return _LIB


def _to_ptr(arr, dtype):
    """Convert numpy array to ctypes pointer, or NULL if None."""
    if arr is None:
        if dtype == np.int8:
            return ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int8))
        return ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float))
    arr = np.ascontiguousarray(arr, dtype=dtype)
    if dtype == np.int8:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ternarize(weight_tensor):
    """Quantize weight tensor to {-1, 0, +1} with TWN threshold."""
    w = weight_tensor.detach().cpu().float().numpy()
    abs_w = np.abs(w)
    delta = 0.7 * abs_w.mean()
    ternary = np.zeros(w.shape, dtype=np.int8)
    mask_pos = w > delta
    mask_neg = w < -delta
    ternary[mask_pos] = 1
    ternary[mask_neg] = -1
    mask_nz = mask_pos | mask_neg
    alpha = float(abs_w[mask_nz].mean()) if mask_nz.any() else float(abs_w.mean())
    sparsity = 1.0 - mask_nz.sum() / ternary.size
    return ternary, alpha, sparsity


def benchmark_ternary_model(model, is_hybrid=False, n_warmup=100, n_runs=1000):
    """Benchmark a PyTorch ternary/hybrid CNN using the C engine.

    Full warmup + benchmark loop runs in C — zero Python overhead.
    Returns dict with latency stats.
    """
    import torch
    model.eval()
    lib = _get_lib()

    state = model.state_dict()
    keys = list(state.keys())

    # Extract conv layers
    conv_weights = []
    conv_alphas = []
    conv_biases = []
    bn_params = []

    # Find conv weight keys (4D tensors)
    conv_keys = [k for k in keys if state[k].dim() == 4]

    for ck in conv_keys:
        tw, alpha, sp = _ternarize(state[ck])
        conv_weights.append(tw)
        conv_alphas.append(alpha)

        # Bias
        bk = ck.replace('weight', 'bias')
        if bk in state and state[bk] is not None:
            conv_biases.append(state[bk].cpu().numpy())
        else:
            conv_biases.append(np.zeros(tw.shape[0], dtype=np.float32))

        # Find BN params after this conv
        # Look for running_mean near this key's position
        idx = keys.index(ck)
        bn = {'gamma': None, 'beta': None, 'mean': None, 'var': None}
        for k2 in keys[idx+1:idx+8]:
            if 'running_mean' in k2:
                prefix = k2.rsplit('.', 1)[0]
                bn['gamma'] = state[f'{prefix}.weight'].cpu().numpy()
                bn['beta'] = state[f'{prefix}.bias'].cpu().numpy()
                bn['mean'] = state[f'{prefix}.running_mean'].cpu().numpy()
                bn['var'] = state[f'{prefix}.running_var'].cpu().numpy()
                break
        bn_params.append(bn)

    # Extract FC layers (2D weight tensors, excluding BN/SE internal)
    fc_keys = [k for k in keys if state[k].dim() == 2
               and 'excite' not in k and 'squeeze' not in k]

    fc_data = []
    for fk in fc_keys:
        w = state[fk]
        bk = fk.replace('weight', 'bias')
        bias = state[bk].cpu().numpy() if bk in state else np.zeros(w.shape[0], dtype=np.float32)

        if is_hybrid:
            fc_data.append({
                'tw': None, 'alpha': 0.0, 'bias': bias,
                'fp_w': w.cpu().numpy(),
                'out': w.shape[0], 'inp': w.shape[1]
            })
        else:
            tw, alpha, sp = _ternarize(w)
            fc_data.append({
                'tw': tw, 'alpha': alpha, 'bias': bias,
                'fp_w': None,
                'out': w.shape[0], 'inp': w.shape[1]
            })

    if len(conv_weights) < 2 or len(fc_data) < 2:
        raise ValueError(f"Expected 2 conv + 2 FC layers, got {len(conv_weights)} conv + {len(fc_data)} FC")

    # Prepare input
    dummy = np.random.randn(1, 28, 28).astype(np.float32)
    latencies = np.zeros(n_runs, dtype=np.float32)

    # Single C call — entire benchmark runs in C
    lib.benchmark_forward(
        # Conv1
        _to_ptr(conv_weights[0], np.int8),
        ctypes.c_float(conv_alphas[0]),
        _to_ptr(conv_biases[0], np.float32),
        _to_ptr(bn_params[0]['gamma'], np.float32),
        _to_ptr(bn_params[0]['beta'], np.float32),
        _to_ptr(bn_params[0]['mean'], np.float32),
        _to_ptr(bn_params[0]['var'], np.float32),
        conv_weights[0].shape[0],
        # Conv2
        _to_ptr(conv_weights[1], np.int8),
        ctypes.c_float(conv_alphas[1]),
        _to_ptr(conv_biases[1], np.float32),
        _to_ptr(bn_params[1]['gamma'], np.float32),
        _to_ptr(bn_params[1]['beta'], np.float32),
        _to_ptr(bn_params[1]['mean'], np.float32),
        _to_ptr(bn_params[1]['var'], np.float32),
        conv_weights[1].shape[0],
        # FC1
        _to_ptr(fc_data[0]['tw'], np.int8),
        ctypes.c_float(fc_data[0]['alpha']),
        _to_ptr(fc_data[0]['bias'], np.float32),
        _to_ptr(fc_data[0]['fp_w'], np.float32),
        fc_data[0]['out'],
        # FC2
        _to_ptr(fc_data[1]['tw'], np.int8),
        ctypes.c_float(fc_data[1]['alpha']),
        _to_ptr(fc_data[1]['bias'], np.float32),
        _to_ptr(fc_data[1]['fp_w'], np.float32),
        fc_data[1]['out'],
        # Input & config
        _to_ptr(dummy, np.float32),
        n_warmup, n_runs,
        _to_ptr(latencies, np.float32),
    )

    latencies = np.sort(latencies)
    result = {
        "avg_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "p99_ms": float(latencies[int(0.99 * n_runs)]),
        "min_ms": float(latencies[0]),
        "max_ms": float(latencies[-1]),
        "n_runs": n_runs,
    }

    # Print sparsity info
    for i, (tw, a) in enumerate(zip(conv_weights, conv_alphas)):
        sp = 1.0 - np.count_nonzero(tw) / tw.size
        print(f"  Conv{i+1}: ch={tw.shape[0]}, alpha={a:.4f}, sparsity={sp:.1%}")
    for i, fd in enumerate(fc_data):
        if fd['tw'] is not None:
            sp = 1.0 - np.count_nonzero(fd['tw']) / fd['tw'].size
            print(f"  FC{i+1}: {fd['inp']}->{fd['out']}, alpha={fd['alpha']:.4f}, sparsity={sp:.1%}")
        else:
            print(f"  FC{i+1}: {fd['inp']}->{fd['out']}, full-precision")

    return result
