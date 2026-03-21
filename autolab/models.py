"""Model registry and built-in architectures."""

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
