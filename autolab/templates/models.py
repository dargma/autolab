"""Project-specific models. Import and register with autolab."""

from autolab.models import register
import torch.nn as nn


# Example: register a custom model type for use in config.yaml.
# Reference it with: type: my_custom_model
#
# @register("my_custom_model")
# class MyModel(nn.Module):
#     def __init__(self, channels, fc, input_size=28, **_kw):
#         super().__init__()
#         # Build your architecture here.
#         # channels: list of channel sizes, e.g. [1, 32, 64]
#         # fc: list of FC layer sizes, e.g. [256, 10]
#         # input_size: spatial dimension of input (28 for MNIST, 32 for CIFAR)
#         ...
#
#     def forward(self, x):
#         ...
#         return x
