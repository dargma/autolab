"""Model registry and built-in architectures."""

import torch
import torch.nn as nn

# ── Registry ──────────────────────────────────────────────────────

_REGISTRY = {}


def register(name):
    """Decorator to register a model class by name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(candidate):
    """Build a model from a candidate config dict.

    Config must have a 'type' key matching a registered name.
    Remaining keys are passed as kwargs to the constructor.
    """
    t = candidate.get("type", "fc")
    if t not in _REGISTRY:
        raise ValueError(f"Unknown model type '{t}'. Registered: {list(_REGISTRY.keys())}")
    kwargs = {k: v for k, v in candidate.items() if k not in ("name", "type")}
    return _REGISTRY[t](**kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── Built-in models ──────────────────────────────────────────────


@register("fc")
class FCNet(nn.Module):
    def __init__(self, layers, **_kw):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


@register("cnn")
class CNNNet(nn.Module):
    def __init__(self, channels, fc, input_size=28, **_kw):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            convs.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.convs = nn.Sequential(*convs)
        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for f in fc:
            fc_mods.extend([nn.Linear(prev, f), nn.ReLU()])
            prev = f
        fc_mods.pop()  # remove last ReLU
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register("cnn_bn")
class CNNBatchNorm(nn.Module):
    """CNN with BatchNorm after each conv — stabilizes training, often +0.5-1% accuracy."""

    def __init__(self, channels, fc, input_size=28, dropout=0.0, **_kw):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            convs.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.convs = nn.Sequential(*convs)
        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for fi, f in enumerate(fc):
            fc_mods.append(nn.Linear(prev, f))
            if fi < len(fc) - 1:
                fc_mods.append(nn.ReLU())
                if dropout > 0:
                    fc_mods.append(nn.Dropout(dropout))
            prev = f
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register("residual_cnn")
class ResidualCNN(nn.Module):
    """CNN with residual (skip) connections — borrowed from ResNet, adapted for tiny models."""

    def __init__(self, channels, fc, input_size=28, **_kw):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.Conv2d(channels[i + 1], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
            )
            self.blocks.append(block)
            # 1x1 conv for channel matching in skip connection
            self.downsamples.append(nn.Conv2d(channels[i], channels[i + 1], 1))
            self.pools.append(nn.MaxPool2d(2))

        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for fi, f in enumerate(fc):
            fc_mods.append(nn.Linear(prev, f))
            if fi < len(fc) - 1:
                fc_mods.append(nn.ReLU())
            prev = f
        self.fc = nn.Sequential(*fc_mods)
        self.relu = nn.ReLU()

    def forward(self, x):
        for block, ds, pool in zip(self.blocks, self.downsamples, self.pools):
            identity = ds(x)
            x = block(x)
            x = self.relu(x + identity)
            x = pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register("squeeze_excite_cnn")
class SqueezeExciteCNN(nn.Module):
    """CNN with Squeeze-and-Excitation blocks — channel attention from SENet (Hu et al. 2018)."""

    def __init__(self, channels, fc, input_size=28, se_ratio=4, **_kw):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            ch_out = channels[i + 1]
            layers.extend([
                nn.Conv2d(channels[i], ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(),
            ])
            # SE block: global pool -> FC -> ReLU -> FC -> Sigmoid -> scale
            se_ch = max(ch_out // se_ratio, 1)
            layers.append(_SEBlock(ch_out, se_ch))
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for fi, f in enumerate(fc):
            fc_mods.append(nn.Linear(prev, f))
            if fi < len(fc) - 1:
                fc_mods.append(nn.ReLU())
            prev = f
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class _SEBlock(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduction),
            nn.ReLU(),
            nn.Linear(reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1)
        return x * w


# ── Ternary Quantization primitives ──────────────────────────────

class _TernaryQuantize(torch.autograd.Function):
    """Ternary weight quantization: w -> {-alpha, 0, +alpha}.

    Uses threshold-based quantization (Li et al., TWN 2016):
      w_t =  alpha  if w > delta
      w_t = -alpha  if w < -delta
      w_t =  0      otherwise
    where delta = 0.7 * mean(|w|), alpha = mean(|w[|w|>delta]|).
    Straight-Through Estimator (STE) for gradient.
    """

    @staticmethod
    def forward(ctx, weight):
        abs_w = weight.abs()
        delta = 0.7 * abs_w.mean()
        mask_pos = weight > delta
        mask_neg = weight < -delta
        mask_nonzero = mask_pos | mask_neg
        alpha = abs_w[mask_nonzero].mean() if mask_nonzero.any() else abs_w.mean()
        ternary = torch.zeros_like(weight)
        ternary[mask_pos] = alpha
        ternary[mask_neg] = -alpha
        ctx.save_for_backward(weight)
        return ternary

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output


def _ternarize(weight):
    return _TernaryQuantize.apply(weight)


class TernaryConv2d(nn.Module):
    """Conv2d with ternary weight quantization during forward pass."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        tw = _ternarize(self.conv.weight)
        return nn.functional.conv2d(x, tw, self.conv.bias,
                                    self.conv.stride, self.conv.padding)


class TernaryLinear(nn.Module):
    """Linear with ternary weight quantization during forward pass."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        tw = _ternarize(self.linear.weight)
        return nn.functional.linear(x, tw, self.linear.bias)


@register("ternary_cnn")
class TernaryCNN(nn.Module):
    """CNN with ternary weight quantization (TWN).

    Weights are constrained to {-alpha, 0, +alpha} during forward pass.
    Trains with full precision via STE, but inference uses ternary weights.
    Cross-domain transplant from model compression / edge-AI literature.
    """

    def __init__(self, channels, fc, input_size=28, **_kw):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            convs.extend([
                TernaryConv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.convs = nn.Sequential(*convs)
        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for fi, f in enumerate(fc):
            fc_mods.append(TernaryLinear(prev, f))
            if fi < len(fc) - 1:
                fc_mods.append(nn.ReLU())
            prev = f
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register("ternary_hybrid_cnn")
class TernaryHybridCNN(nn.Module):
    """Hybrid: ternary conv weights + full-precision FC head.

    Rationale: conv layers dominate compute, FC head dominates accuracy.
    Ternarizing only convs gives most of the speed benefit with less accuracy loss.
    """

    def __init__(self, channels, fc, input_size=28, **_kw):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            convs.extend([
                TernaryConv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.convs = nn.Sequential(*convs)
        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for fi, f in enumerate(fc):
            fc_mods.append(nn.Linear(prev, f))  # full precision FC
            if fi < len(fc) - 1:
                fc_mods.append(nn.ReLU())
            prev = f
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register("depthwise")
class DepthwiseCNN(nn.Module):
    def __init__(self, channels, fc, input_size=28, **_kw):
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
        spatial = input_size // (2 ** (len(channels) - 1))
        fc_in = channels[-1] * spatial * spatial
        fc_mods = []
        prev = fc_in
        for f in fc:
            fc_mods.extend([nn.Linear(prev, f), nn.ReLU()])
            prev = f
        fc_mods.pop()
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
