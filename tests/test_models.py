"""Tests for autolab.models — model registry, build, forward pass, param count."""

import pytest
import torch

from autolab.models import _REGISTRY, build_model, count_params


# Configs that produce valid models for each registered type.
# All use 1-channel 28x28 input and 10-class output.
MODEL_CONFIGS = {
    "fc": {"type": "fc", "layers": [784, 128, 10]},
    "cnn": {"type": "cnn", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
    "cnn_bn": {"type": "cnn_bn", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
    "residual_cnn": {"type": "residual_cnn", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
    "squeeze_excite_cnn": {"type": "squeeze_excite_cnn", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
    "ternary_cnn": {"type": "ternary_cnn", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
    "ternary_hybrid_cnn": {"type": "ternary_hybrid_cnn", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
    "depthwise": {"type": "depthwise", "channels": [1, 16], "fc": [256, 10], "input_size": 28},
}


class TestModelRegistry:
    """Ensure every registered model can be built, run forward, and counted."""

    @pytest.mark.parametrize("model_type", list(_REGISTRY.keys()))
    def test_forward_pass_shape(self, model_type):
        """Each model should accept (2, 1, 28, 28) and return (2, 10)."""
        config = MODEL_CONFIGS[model_type]
        model = build_model(config)
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10), f"{model_type}: expected (2, 10), got {out.shape}"

    @pytest.mark.parametrize("model_type", list(_REGISTRY.keys()))
    def test_count_params_positive(self, model_type):
        """count_params should return a positive integer for every model."""
        config = MODEL_CONFIGS[model_type]
        model = build_model(config)
        n = count_params(model)
        assert isinstance(n, int)
        assert n > 0, f"{model_type}: param count should be positive, got {n}"

    def test_build_model_unknown_type_raises(self):
        """build_model with an unregistered type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model({"type": "nonexistent_model_xyz"})

    def test_all_registered_types_have_configs(self):
        """Safety check: MODEL_CONFIGS covers every registered type."""
        for name in _REGISTRY:
            assert name in MODEL_CONFIGS, (
                f"Registered model '{name}' has no test config in MODEL_CONFIGS"
            )
