"""Ternary v2 engine — adder-only inference with bit-packed weights + int16 activations.

No multiplier in the datapath. Conv/FC are pure add/sub on int16 (Q8.8).
Only 2 float ops per output element (alpha scale + bias add).
"""

import ctypes
import subprocess
import time
import numpy as np
from pathlib import Path

_LIB_DIR = Path(__file__).parent / "csrc"
_LIB_PATH = _LIB_DIR / "libternary_v2.so"
_LIB = None


def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    src = _LIB_DIR / "ternary_v2.c"
    if not _LIB_PATH.exists() or _LIB_PATH.stat().st_mtime < src.stat().st_mtime:
        print("Compiling ternary_v2 (adder-only engine)...")
        subprocess.run([
            "gcc", "-O3", "-march=native", "-ffast-math", "-funroll-loops",
            "-shared", "-fPIC", "-o", str(_LIB_PATH), str(src), "-lm"
        ], check=True)
        print(f"Compiled: {_LIB_PATH}")

    _LIB = ctypes.CDLL(str(_LIB_PATH))

    # benchmark_v2 signature
    _LIB.benchmark_v2.argtypes = [
        # Conv1: pos, neg, alpha, bias, bn_scale, bn_shift, out_ch
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        # Conv2
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        # FC1: pos, neg, alpha, bias, fpw, out
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        # FC2
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        # Input, is_hybrid, warmup, runs
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        # Output
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    _LIB.benchmark_v2.restype = None
    return _LIB


def _ptr(arr, dtype=np.float32):
    if arr is None:
        if dtype == np.uint8:
            return ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8))
        return ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
    arr = np.ascontiguousarray(arr, dtype=dtype)
    if dtype == np.uint8:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    elif dtype == np.int32:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ternarize_and_pack(weight_tensor):
    """Ternarize weights and pack into pos/neg bitmasks.

    Returns (pos_mask, neg_mask, alpha, sparsity).
    """
    w = weight_tensor.detach().cpu().float().numpy()
    abs_w = np.abs(w)
    delta = 0.7 * abs_w.mean()

    flat = w.flatten()
    n = len(flat)
    n_bytes = (n + 7) // 8

    pos_mask = np.zeros(n_bytes, dtype=np.uint8)
    neg_mask = np.zeros(n_bytes, dtype=np.uint8)

    n_nonzero = 0
    sum_abs = 0.0
    for i in range(n):
        byte_idx = i // 8
        bit_idx = i % 8
        if flat[i] > delta:
            pos_mask[byte_idx] |= (1 << bit_idx)
            n_nonzero += 1
            sum_abs += abs(flat[i])
        elif flat[i] < -delta:
            neg_mask[byte_idx] |= (1 << bit_idx)
            n_nonzero += 1
            sum_abs += abs(flat[i])

    alpha = sum_abs / n_nonzero if n_nonzero > 0 else float(abs_w.mean())
    sparsity = 1.0 - n_nonzero / n
    return pos_mask, neg_mask, alpha, sparsity


def benchmark_model(model, is_hybrid=False, n_warmup=200, n_runs=2000):
    """Benchmark a PyTorch ternary CNN using the adder-only C engine.

    Returns dict with latency stats + sparsity info.
    """
    import torch
    model.eval()
    lib = _get_lib()
    state = model.state_dict()
    keys = list(state.keys())

    # Find conv weight keys (4D tensors)
    conv_keys = [k for k in keys if state[k].dim() == 4]
    if len(conv_keys) < 2:
        raise ValueError(f"Need 2 conv layers, found {len(conv_keys)}")

    conv_data = []
    for ck in conv_keys[:2]:
        w = state[ck]
        pos, neg, alpha, sp = _ternarize_and_pack(w)

        # Bias
        bk = ck.replace('weight', 'bias')
        bias = state[bk].cpu().numpy() if bk in state else np.zeros(w.shape[0], dtype=np.float32)

        # BN params — find running_mean near this conv
        idx = keys.index(ck)
        bn_scale = np.ones(w.shape[0], dtype=np.float32)
        bn_shift = np.zeros(w.shape[0], dtype=np.float32)
        for k2 in keys[idx+1:idx+8]:
            if 'running_mean' in k2:
                prefix = k2.rsplit('.', 1)[0]
                gamma = state[f'{prefix}.weight'].cpu().numpy()
                beta = state[f'{prefix}.bias'].cpu().numpy()
                mean = state[f'{prefix}.running_mean'].cpu().numpy()
                var = state[f'{prefix}.running_var'].cpu().numpy()
                bn_scale = gamma / np.sqrt(var + 1e-5)
                bn_shift = beta - mean * bn_scale
                break

        conv_data.append({
            'pos': pos, 'neg': neg, 'alpha': alpha, 'bias': bias,
            'bn_scale': bn_scale.astype(np.float32),
            'bn_shift': bn_shift.astype(np.float32),
            'out_ch': w.shape[0], 'sparsity': sp
        })

    # FC layers (2D weights, skip BN/SE internals)
    fc_keys = [k for k in keys if state[k].dim() == 2
               and 'excite' not in k and 'squeeze' not in k]

    fc_data = []
    for fk in fc_keys[:2]:
        w = state[fk]
        bk = fk.replace('weight', 'bias')
        bias = state[bk].cpu().numpy() if bk in state else np.zeros(w.shape[0], dtype=np.float32)

        if is_hybrid:
            fc_data.append({
                'pos': np.zeros(1, dtype=np.uint8), 'neg': np.zeros(1, dtype=np.uint8),
                'alpha': 0.0, 'bias': bias,
                'fpw': w.cpu().numpy().astype(np.float32),
                'out': w.shape[0], 'sparsity': 0.0
            })
        else:
            pos, neg, alpha, sp = _ternarize_and_pack(w)
            fc_data.append({
                'pos': pos, 'neg': neg, 'alpha': alpha, 'bias': bias,
                'fpw': None, 'out': w.shape[0], 'sparsity': sp
            })

    if len(fc_data) < 2:
        raise ValueError(f"Need 2 FC layers, found {len(fc_data)}")

    # Input
    dummy = np.random.randn(28 * 28).astype(np.float32) * 0.3  # normalized range
    latencies = np.zeros(n_runs, dtype=np.float32)
    prediction = np.zeros(1, dtype=np.int32)

    # Single C call — entire benchmark in C, zero Python overhead
    lib.benchmark_v2(
        _ptr(conv_data[0]['pos'], np.uint8), _ptr(conv_data[0]['neg'], np.uint8),
        ctypes.c_float(conv_data[0]['alpha']), _ptr(conv_data[0]['bias']),
        _ptr(conv_data[0]['bn_scale']), _ptr(conv_data[0]['bn_shift']),
        conv_data[0]['out_ch'],

        _ptr(conv_data[1]['pos'], np.uint8), _ptr(conv_data[1]['neg'], np.uint8),
        ctypes.c_float(conv_data[1]['alpha']), _ptr(conv_data[1]['bias']),
        _ptr(conv_data[1]['bn_scale']), _ptr(conv_data[1]['bn_shift']),
        conv_data[1]['out_ch'],

        _ptr(fc_data[0]['pos'], np.uint8), _ptr(fc_data[0]['neg'], np.uint8),
        ctypes.c_float(fc_data[0]['alpha']), _ptr(fc_data[0]['bias']),
        _ptr(fc_data[0]['fpw']), fc_data[0]['out'],

        _ptr(fc_data[1]['pos'], np.uint8), _ptr(fc_data[1]['neg'], np.uint8),
        ctypes.c_float(fc_data[1]['alpha']), _ptr(fc_data[1]['bias']),
        _ptr(fc_data[1]['fpw']), fc_data[1]['out'],

        _ptr(dummy), ctypes.c_int(1 if is_hybrid else 0),
        n_warmup, n_runs,
        _ptr(latencies),
        prediction.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    latencies = np.sort(latencies)

    # Print info
    for i, cd in enumerate(conv_data):
        print(f"  Conv{i+1}: ch={cd['out_ch']}, alpha={cd['alpha']:.4f}, "
              f"sparsity={cd['sparsity']:.1%} (zeros={cd['sparsity']:.0%} skipped)")
    for i, fd in enumerate(fc_data):
        if fd['fpw'] is not None:
            print(f"  FC{i+1}: {fd['out']} out, full-precision (hybrid)")
        else:
            print(f"  FC{i+1}: {fd['out']} out, alpha={fd['alpha']:.4f}, "
                  f"sparsity={fd['sparsity']:.1%}")

    return {
        "avg_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "p99_ms": float(latencies[int(0.99 * n_runs)]),
        "min_ms": float(latencies[0]),
        "n_runs": n_runs,
    }
