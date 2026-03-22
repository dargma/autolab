"""Ternary inference engine — extracts ternary weights from PyTorch, runs C kernel.

Usage:
    from autolab.ternary_engine import TernaryEngine
    engine = TernaryEngine()
    engine.load_from_pytorch(model)  # extracts & ternarizes weights
    latency_ms = engine.benchmark()  # runs C inference, measures latency
    pred = engine.predict(input_28x28)  # single-sample inference
"""

import ctypes
import os
import time
import subprocess
import numpy as np
from pathlib import Path

_LIB_DIR = Path(__file__).parent / "csrc"
_LIB_PATH = _LIB_DIR / "libternary.so"


def _compile_if_needed():
    """Compile the C ternary inference library if not already built."""
    src = _LIB_DIR / "ternary_inference.c"
    if not src.exists():
        raise FileNotFoundError(f"C source not found: {src}")

    # Recompile if source is newer than library
    if _LIB_PATH.exists() and _LIB_PATH.stat().st_mtime > src.stat().st_mtime:
        return

    cmd = [
        "gcc", "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC",
        "-o", str(_LIB_PATH),
        str(src), "-lm"
    ]
    print(f"Compiling ternary inference engine...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    print(f"Compiled: {_LIB_PATH}")


def _ternarize_weights(weight_tensor):
    """Quantize a weight tensor to {-1, 0, +1} with scaling factor alpha.

    Uses TWN threshold: delta = 0.7 * mean(|w|)
    Returns (int8_weights, alpha).
    """
    import torch
    w = weight_tensor.detach().cpu().float()
    abs_w = w.abs()
    delta = 0.7 * abs_w.mean().item()

    ternary = np.zeros(w.shape, dtype=np.int8)
    w_np = w.numpy()
    mask_pos = w_np > delta
    mask_neg = w_np < -delta
    ternary[mask_pos] = 1
    ternary[mask_neg] = -1

    # Alpha = mean of absolute values where non-zero
    mask_nonzero = mask_pos | mask_neg
    if mask_nonzero.any():
        alpha = float(np.abs(w_np[mask_nonzero]).mean())
    else:
        alpha = float(abs_w.mean().item())

    sparsity = 1.0 - mask_nonzero.sum() / ternary.size
    return ternary, alpha, sparsity


class TernaryEngine:
    """Python interface to the C ternary inference engine."""

    def __init__(self):
        _compile_if_needed()
        self._lib = ctypes.CDLL(str(_LIB_PATH))

        # Bind C functions
        self._lib.ternary_conv2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # input
            ctypes.POINTER(ctypes.c_int8),    # weights
            ctypes.POINTER(ctypes.c_float),   # bias
            ctypes.c_float,                    # alpha
            ctypes.POINTER(ctypes.c_float),   # output
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # in_ch, H, W
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_ch, kH, kW
            ctypes.c_int, ctypes.c_int,        # pad, stride
        ]

        self._lib.ternary_linear.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # input
            ctypes.POINTER(ctypes.c_int8),    # weights
            ctypes.POINTER(ctypes.c_float),   # bias
            ctypes.c_float,                    # alpha
            ctypes.POINTER(ctypes.c_float),   # output
            ctypes.c_int, ctypes.c_int,        # in_features, out_features
        ]

        self._lib.fp_linear.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int,
        ]

        self._lib.batchnorm2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self._lib.relu_inplace.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int]

        self._lib.maxpool2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self._lib.argmax.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self._lib.argmax.restype = ctypes.c_int

        # Model weights storage
        self._layers = []
        self._is_hybrid = False
        self._num_classes = 10

    def _to_c_float(self, arr):
        """Convert numpy array to ctypes float pointer."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr

    def _to_c_int8(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.int8)
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), arr

    def load_from_pytorch(self, model, is_hybrid=False):
        """Extract weights from a trained PyTorch ternary/hybrid CNN.

        Supports: TernaryCNN, TernaryHybridCNN, CNNBatchNorm, CNNNet
        Architecture assumed: [Conv+BN+ReLU+Pool]xN -> FC+ReLU -> FC
        """
        import torch
        model.eval()
        self._is_hybrid = is_hybrid
        self._layers = []

        state = model.state_dict()
        keys = list(state.keys())

        # Parse conv layers
        conv_idx = 0
        i = 0
        conv_layers = []
        while i < len(keys):
            k = keys[i]
            # Look for conv weight patterns
            if 'conv' in k and 'weight' in k and state[k].dim() == 4:
                layer = {}
                # Get conv weight
                w = state[k]
                tw, alpha, sparsity = _ternarize_weights(w)
                layer['type'] = 'conv'
                layer['tw'] = tw
                layer['alpha'] = alpha
                layer['sparsity'] = sparsity
                layer['out_ch'] = tw.shape[0]
                layer['in_ch'] = tw.shape[1]
                layer['kH'] = tw.shape[2]
                layer['kW'] = tw.shape[3]

                # Find matching bias
                bias_key = k.replace('weight', 'bias')
                if bias_key in state and state[bias_key] is not None:
                    layer['bias'] = state[bias_key].detach().cpu().numpy()
                else:
                    layer['bias'] = np.zeros(tw.shape[0], dtype=np.float32)

                # Look for BatchNorm after this conv
                bn_prefix = None
                for bk in keys[i+1:i+10]:
                    if 'running_mean' in bk:
                        bn_prefix = bk.rsplit('.', 1)[0]
                        break
                if bn_prefix:
                    layer['bn'] = True
                    layer['bn_gamma'] = state[f'{bn_prefix}.weight'].detach().cpu().numpy()
                    layer['bn_beta'] = state[f'{bn_prefix}.bias'].detach().cpu().numpy()
                    layer['bn_mean'] = state[f'{bn_prefix}.running_mean'].detach().cpu().numpy()
                    layer['bn_var'] = state[f'{bn_prefix}.running_var'].detach().cpu().numpy()
                else:
                    layer['bn'] = False

                conv_layers.append(layer)
                conv_idx += 1
            i += 1

        # Parse FC layers
        fc_layers = []
        for i, k in enumerate(keys):
            if state[k].dim() == 2:  # weight matrix
                layer = {}
                w = state[k]
                bias_key = k.replace('weight', 'bias')

                if is_hybrid or 'linear' not in k.lower():
                    # Check if this is a ternary or FP layer
                    # For hybrid: FC layers are full precision
                    if is_hybrid:
                        layer['type'] = 'fp_linear'
                        layer['w'] = w.detach().cpu().numpy()
                    else:
                        layer['type'] = 'ternary_linear'
                        tw, alpha, sparsity = _ternarize_weights(w)
                        layer['tw'] = tw
                        layer['alpha'] = alpha
                        layer['sparsity'] = sparsity

                    if not is_hybrid:
                        tw, alpha, sparsity = _ternarize_weights(w)
                        layer['type'] = 'ternary_linear'
                        layer['tw'] = tw
                        layer['alpha'] = alpha
                        layer['sparsity'] = sparsity
                else:
                    layer['type'] = 'fp_linear'
                    layer['w'] = w.detach().cpu().numpy()

                layer['in_features'] = w.shape[1]
                layer['out_features'] = w.shape[0]

                if bias_key in state:
                    layer['bias'] = state[bias_key].detach().cpu().numpy()
                else:
                    layer['bias'] = np.zeros(w.shape[0], dtype=np.float32)

                fc_layers.append(layer)

        self._conv_layers = conv_layers
        self._fc_layers = fc_layers
        if fc_layers:
            self._num_classes = fc_layers[-1]['out_features']

        total_sparsity = []
        for cl in conv_layers:
            total_sparsity.append(cl['sparsity'])
            print(f"  Conv {cl['in_ch']}->{cl['out_ch']}: alpha={cl['alpha']:.4f}, "
                  f"sparsity={cl['sparsity']:.1%}")
        for fl in fc_layers:
            if fl['type'] == 'ternary_linear':
                total_sparsity.append(fl['sparsity'])
                print(f"  FC {fl['in_features']}->{fl['out_features']}: "
                      f"alpha={fl['alpha']:.4f}, sparsity={fl['sparsity']:.1%}")
            else:
                print(f"  FC {fl['in_features']}->{fl['out_features']}: full-precision")

        if total_sparsity:
            print(f"  Average ternary sparsity: {np.mean(total_sparsity):.1%}")

    def predict(self, input_np):
        """Run inference on a single 28x28 input. Returns class index."""
        assert input_np.shape == (1, 28, 28) or input_np.shape == (28, 28)
        if input_np.ndim == 2:
            input_np = input_np.reshape(1, 28, 28)

        x = np.ascontiguousarray(input_np, dtype=np.float32)
        H, W = 28, 28

        buf_a = np.zeros(64 * 28 * 28, dtype=np.float32)
        buf_b = np.zeros(64 * 28 * 28, dtype=np.float32)

        # Conv layers
        for cl in self._conv_layers:
            tw_p, tw_arr = self._to_c_int8(cl['tw'])
            bias_p, bias_arr = self._to_c_float(cl['bias'])
            x_p, x_arr = self._to_c_float(x)
            out = np.zeros(cl['out_ch'] * H * W, dtype=np.float32)
            out_p, out_arr = self._to_c_float(out)

            self._lib.ternary_conv2d(
                x_p, tw_p, bias_p, ctypes.c_float(cl['alpha']),
                out_p,
                cl['in_ch'], H, W, cl['out_ch'], cl['kH'], cl['kW'],
                1, 1  # pad=1, stride=1
            )

            if cl['bn']:
                g_p, g = self._to_c_float(cl['bn_gamma'])
                b_p, b = self._to_c_float(cl['bn_beta'])
                m_p, m = self._to_c_float(cl['bn_mean'])
                v_p, v = self._to_c_float(cl['bn_var'])
                self._lib.batchnorm2d(out_p, g_p, b_p, m_p, v_p,
                                      ctypes.c_float(1e-5),
                                      cl['out_ch'], H, W)

            self._lib.relu_inplace(out_p, cl['out_ch'] * H * W)

            pool_out = np.zeros(cl['out_ch'] * (H // 2) * (W // 2), dtype=np.float32)
            pool_p, pool_arr = self._to_c_float(pool_out)
            self._lib.maxpool2d(out_p, pool_p, cl['out_ch'], H, W)
            H //= 2
            W //= 2
            x = pool_arr[:cl['out_ch'] * H * W]

        # FC layers
        for fi, fl in enumerate(self._fc_layers):
            x_p, x_arr = self._to_c_float(x)
            out = np.zeros(fl['out_features'], dtype=np.float32)
            out_p, out_arr = self._to_c_float(out)
            bias_p, bias_arr = self._to_c_float(fl['bias'])

            if fl['type'] == 'ternary_linear':
                tw_p, tw_arr = self._to_c_int8(fl['tw'])
                self._lib.ternary_linear(
                    x_p, tw_p, bias_p, ctypes.c_float(fl['alpha']),
                    out_p, fl['in_features'], fl['out_features']
                )
            else:
                w_p, w_arr = self._to_c_float(fl['w'])
                self._lib.fp_linear(
                    x_p, w_p, bias_p, out_p,
                    fl['in_features'], fl['out_features']
                )

            # ReLU between FC layers (not after last)
            if fi < len(self._fc_layers) - 1:
                self._lib.relu_inplace(out_p, fl['out_features'])

            x = out_arr

        return self._lib.argmax(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._num_classes)

    def benchmark(self, n_warmup=50, n_runs=500):
        """Benchmark latency with a random 28x28 input.

        Returns dict with avg_ms, p99_ms, min_ms.
        """
        dummy = np.random.randn(1, 28, 28).astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            self.predict(dummy)

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            t1 = time.perf_counter()
            self.predict(dummy)
            t2 = time.perf_counter()
            latencies.append((t2 - t1) * 1000)

        latencies.sort()
        return {
            "avg_ms": sum(latencies) / len(latencies),
            "p99_ms": latencies[int(0.99 * len(latencies))],
            "min_ms": latencies[0],
            "median_ms": latencies[len(latencies) // 2],
        }

    def evaluate_accuracy(self, test_loader):
        """Evaluate accuracy on a PyTorch test DataLoader."""
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            for i in range(batch_x.shape[0]):
                x = batch_x[i].numpy()  # [1, 28, 28]
                pred = self.predict(x)
                if pred == batch_y[i].item():
                    correct += 1
                total += 1
        return correct / total
