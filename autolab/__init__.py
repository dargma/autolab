"""Autolab — Autonomous AI Research Framework."""
__version__ = "0.2.0"

from .models import register, build_model, count_params
from .data import get_loaders, get_info, DATASETS
from .config import load_goal, load_sweep_config
from .safety import check_disk
