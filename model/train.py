"""Compatibility shim for the uploaded ``train(1).py`` source file.

The uploaded filename is retained, while ``evaluate.py`` can continue to use
the conventional ``from train import ...`` imports.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

_source = MODEL_DIR / "train(1).py"
_spec = importlib.util.spec_from_file_location("ardn_training_source", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

train = _module.train
forward_and_losses = _module.forward_and_losses
total_loss = _module.total_loss
