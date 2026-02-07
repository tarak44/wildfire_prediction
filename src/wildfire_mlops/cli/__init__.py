"""CLI entrypoints."""

from .main import main as inference_main
from .train import main as train_main

__all__ = ["inference_main", "train_main"]
