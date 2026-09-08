"""Adapter between the live demo simulator and the trained ARDN predictor.

ARDN was trained on a synthetic, topology-invariant graph representation.  The
live demo has one node per echelon, so this module maps its current state into
the same 25-feature graph schema and ranks ARDN's six candidate interventions.
It is advisory only: the deterministic verifier remains the execution gate.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import pickle
import sys

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
ACTION_LABELS = {
    "reallocate": "Reallocate inventory",
    "reroute": "Reroute shipment",
    "delay": "Delay demand commitment",
    "prioritize": "Prioritize constrained node",
    "share_capacity": "Share capacity",
    "emergency_source": "Use emergency source",
}
DISRUPTION_TYPE_MAP = {
    "supplier_capacity_drop": "capacity",
    "route_closure": "transport",
    "demand_spike": "demand",
}
DISRUPTION_TARGET = {
    "supplier_capacity_drop": 0,
    "route_closure": 1,
    "demand_spike": 3,
}


class ARDNUnavailable(RuntimeError):
    """The optional local ARDN artifact cannot be loaded."""


@lru_cache(maxsize=1)
def _load_ardn() -> tuple[Any, tuple[Any, Any, Any], list[str]]:
    """Load the user-provided, local trained artifact once per API process."""
    artifact = MODEL_DIR / "trained_model.pkl"
    if not artifact.is_file():
        raise ARDNUnavailable(f"Missing trained model at {artifact}")
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    try:
        # These modules intentionally use local absolute imports (autograd,
        # simulator), hence MODEL_DIR is placed first in sys.path above.
        from model import ARDN  # type: ignore
        from simulator import ACTION_TYPES, default_network  # type: ignore
        # The artifact is part of this user-provided local project. Do not use
        # this loader with a model file from an untrusted source.
        with artifact.open("rb") as file:
            saved = pickle.load(file)
        model = ARDN(horizon=saved["horizon"], tau=saved["tau"])
        model.load_state_dict(saved["state_dict"])
        model.ood_mean = saved["ood_mean"]
        model.ood_cov_inv = saved["ood_cov_inv"]
        return model, default_network(n_per_echelon=1), list(ACTION_TYPES)
    except Exception as exc:  # pragma: no cover - exposed as a usable API state
        raise ARDNUnavailable(f"Could not load ARDN: {exc}") from exc


def _episode_from_engine(engine: Any, action_type: str, net: tuple[Any, Any, Any]) -> dict[str, Any]:
    """Represent the live engine state using ARDN's trained feature contract."""
    from simulator import SupplyChainSim  # loaded from MODEL_DIR by _load_ardn

    node_types, edges, echelons = net
    disruption = engine.disruption
    target = DISRUPTION_TARGET[disruption.type]
    action = {
        "type": action_type,
        "source": max(0, target - 1),
        "target": target,
        "magnitude": float(np.clip(disruption.severity, 0.2, 1.0)),
        "duration": int(np.clip(disruption.duration, 2, 6)),
    }
    ardn_disruption = {
        "nodes": [target],
        "severity": float(disruption.severity),
        "duration": int(np.clip(disruption.duration, 2, 8)),
        "type": DISRUPTION_TYPE_MAP[disruption.type],
        "cooccurrence": False,
    }
    sim = SupplyChainSim(node_types, edges, echelons, rng=np.random.default_rng(engine.seed))
    sim.reset(disruption=ardn_disruption, action=action)

    ordered_nodes = [engine.nodes[key] for key in ("supplier", "manufacturer", "distributor", "retailer")]
    sim.inventory = np.array([node.inventory for node in ordered_nodes], dtype=float)
    sim.capacity = np.maximum(np.array([node.capacity for node in ordered_nodes], dtype=float), 1.0)
    sim.demand = np.full(4, max(float(engine.last_demand), 1.0))
    sim.unit_cost = np.array([node.holding_cost + node.shortage_cost / 4 for node in ordered_nodes], dtype=float)
    sim.contract_floor = np.array([node.service_level_target for node in ordered_nodes], dtype=float)
    sim.service = np.clip(
        np.array([min(1.0, node.inventory / max(node.capacity, 1.0)) for node in ordered_nodes], dtype=float),
        0.05,
        1.0,
    )
    # Preserve the observed retailer service level, which is the live
    # simulator's explicit service metric.
    sim.service[-1] = float(np.clip(ordered_nodes[-1].service_level, 0.05, 1.0))
    sim.backlog = np.maximum(0.0, sim.demand * (1.0 - sim.service))

    edge_index, edge_feat = sim.build_edge_index_and_features()
    d_vec, a_vec = sim.build_global_descriptors()
    return {
        "node_types": np.array(node_types),
        "X": sim.build_node_features(),
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "d_vec": d_vec,
        "a_vec": a_vec,
        "hist": sim.history_summary(),
    }


def _number(tensor: Any) -> float:
    return float(tensor.data.reshape(-1)[0])


def _cost_interval(cost: tuple[Any, Any, Any, Any]) -> tuple[float, float]:
    mu, v, alpha, beta = (_number(item) for item in cost)
    variance = beta / (v * (alpha - 1.0) + 1e-6)
    sigma = max(variance, 1e-3) ** 0.5 * 20.0  # cost target was trained divided by 20
    point = mu * 20.0
    return point - sigma, point + sigma


def forecast_for_engine(engine: Any) -> dict[str, Any]:
    """Rank candidate actions and expose ARDN's forecast in JSON-safe values."""
    try:
        model, net, action_types = _load_ardn()
        forecasts = []
        for action_type in action_types:
            episode = _episode_from_engine(engine, action_type, net)
            prediction = model.predict(episode)
            lower, upper = _cost_interval(prediction["cost"])
            recovery_variance = max(0.0, _number(prediction["T_rec_var"]))
            trajectory = [round(float(step.data.mean()), 3) for step in prediction["s_hat_seq"]]
            forecasts.append({
                "action": action_type,
                "label": ACTION_LABELS[action_type],
                "recovery_days": round(max(0.0, _number(prediction["T_rec"])), 1),
                "recovery_uncertainty_days": round(recovery_variance ** 0.5, 1),
                "predicted_cost": round(_number(prediction["cost"][0]) * 20.0, 1),
                "cost_interval": [round(lower, 1), round(upper, 1)],
                "service_loss": round(max(0.0, _number(prediction["L_service"])), 3),
                "risk_probability": round(float(np.clip(_number(prediction["risk_p"]), 0.0, 1.0)), 3),
                "trajectory": trajectory,
                "ood_score": round(model.ood_score(episode), 2),
            })
        forecasts.sort(key=lambda item: (item["recovery_days"], item["predicted_cost"]))
        recommended = forecasts[0]
        return {
            "status": "ready",
            "model": "ARDN",
            "horizon_days": model.horizon,
            "recommendation": recommended,
            "alternatives": forecasts,
            "note": "ARDN is an advisory forecast. The deterministic verifier remains the execution gate.",
        }
    except ARDNUnavailable as exc:
        return {"status": "unavailable", "model": "ARDN", "message": str(exc)}
