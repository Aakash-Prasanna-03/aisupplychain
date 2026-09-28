"""
Model loading and training orchestration for ARDN and baselines.
Provides load_model used by diagnostics.py and comparative evaluations.
"""
from pathlib import Path
import pickle
import numpy as np

from model import ARDN
from baselines import (
    OracleBaseline, RandomForestBaseline, FlatMLP, GNNBaseline,
    train_flatmlp, train_gnn,
)
from dataset import load_all, build_all

MODEL_DIR = Path(__file__).resolve().parent
_CACHE = {}


def _get_dataset():
    data_path = MODEL_DIR / "dataset.pkl"
    if not data_path.is_file():
        return build_all(str(data_path))
    return load_all(str(data_path))


def load_model(name: str, cache: bool = True):
    """
    Loads or instantiates a model by its standard name.
    Supports caching so repeated calls in evaluation loops are instant.
    """
    if cache and name in _CACHE:
        return _CACHE[name]

    model = None

    if name == "Oracle":
        model = OracleBaseline(K=40, seed=0)

    elif name == "RandomForest":
        ckpt = MODEL_DIR / "rf_model.pkl"
        if ckpt.is_file():
            with open(ckpt, "rb") as f:
                model = pickle.load(f)
        else:
            data = _get_dataset()
            model = RandomForestBaseline(horizon=25, seed=0)
            model.fit(data["train"])
            try:
                with open(ckpt, "wb") as f:
                    pickle.dump(model, f)
            except Exception:
                pass

    elif name == "FlatMLP":
        ckpt = MODEL_DIR / "flatmlp_model.pkl"
        if ckpt.is_file():
            with open(ckpt, "rb") as f:
                saved = pickle.load(f)
            model = FlatMLP(horizon=saved.get("horizon", 25))
            model.load_state_dict(saved["state_dict"])
            model.ood_mean = saved.get("ood_mean")
            model.ood_cov_inv = saved.get("ood_cov_inv")
        else:
            data = _get_dataset()
            model = train_flatmlp(data["train"], horizon=25, n_epochs=15, seed=0)
            try:
                with open(ckpt, "wb") as f:
                    pickle.dump({
                        "horizon": 25,
                        "state_dict": model.state_dict(),
                        "ood_mean": getattr(model, "ood_mean", None),
                        "ood_cov_inv": getattr(model, "ood_cov_inv", None),
                    }, f)
            except Exception:
                pass

    elif name == "GNN":
        ckpt = MODEL_DIR / "gnn_model.pkl"
        if ckpt.is_file():
            with open(ckpt, "rb") as f:
                saved = pickle.load(f)
            model = GNNBaseline(horizon=saved.get("horizon", 25))
            model.load_state_dict(saved["state_dict"])
            model.ood_mean = saved.get("ood_mean")
            model.ood_cov_inv = saved.get("ood_cov_inv")
        else:
            data = _get_dataset()
            model = train_gnn(data["train"], horizon=25, n_epochs=15, seed=0)
            try:
                with open(ckpt, "wb") as f:
                    pickle.dump({
                        "horizon": 25,
                        "state_dict": model.state_dict(),
                        "ood_mean": getattr(model, "ood_mean", None),
                        "ood_cov_inv": getattr(model, "ood_cov_inv", None),
                    }, f)
            except Exception:
                pass

    elif name in ("ARDN_v1", "ARDN_v2_full", "ARDN_no_residual", "ARDN_no_graph", "ARDN_no_action"):
        # Check specific artifact first, fall back to trained_model.pkl
        spec_ckpt = MODEL_DIR / f"{name.lower()}.pkl"
        default_ckpt = MODEL_DIR / "trained_model.pkl"
        artifact = spec_ckpt if spec_ckpt.is_file() else default_ckpt

        if artifact.is_file():
            with open(artifact, "rb") as f:
                saved = pickle.load(f)
            use_gat = False if name == "ARDN_no_graph" else True
            model = ARDN(horizon=saved.get("horizon", 15), tau=saved.get("tau", 0.9), use_gat=use_gat)
            model.load_state_dict(saved["state_dict"])
            model.ood_mean = saved.get("ood_mean")
            model.ood_cov_inv = saved.get("ood_cov_inv")
        else:
            model = ARDN(horizon=15, tau=0.9)

    else:
        raise ValueError(f"Unknown model name: {name}")

    if cache and model is not None:
        _CACHE[name] = model

    return model
