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
RUNTIME_TUNING = {
    "forecast_horizon": 15,
    "service_threshold": 0.9,
    "cost_interval_multiplier": 1.0,
    "recovery_weight": 1.0,
    "cost_weight": 0.35,
    "risk_weight": 0.75,
    "service_loss_weight": 0.75,
    "ood_weight": 0.35,
    "risk_review_threshold": 0.6,
    "ood_review_threshold": 15.0,
    "enabled_actions": list(ACTION_LABELS),
}
def _ardn_profile(disruption: Any, physical_nodes: list[Any] | None = None) -> tuple[str, int]:
    """Project arbitrary operational effects onto ARDN's trained coarse inputs."""
    kinds = {effect.type.lower().replace("-", "_") for effect in disruption.effects}
    if any(kind in kinds for kind in {"demand_increase", "demand_spike", "demand_change"}):
        return "demand", 3
    if any(kind in kinds for kind in {"route_closure", "shipping_delay", "transport_disruption"}):
        return "transport", 1
    target = (disruption.affected_nodes or ["supplier"])[0]
    physical_by_id = {node.id: node for node in (physical_nodes or []) if getattr(node, "id", None)}
    if target in physical_by_id:
        target = physical_by_id[target].operating_tier
    return "capacity", {"supplier": 0, "manufacturer": 1, "distributor": 2, "retailer": 3}.get(target, 0)

def _scenario_features(engine: Any) -> dict[str, Any]:
    effects = engine.disruption.effects
    def magnitude(kinds: set[str]) -> float:
        return round(max((effect.magnitude for effect in effects if effect.type in kinds), default=0.0), 3)
    return {
        "affected_nodes": list(engine.disruption.affected_nodes),
        "affected_routes": [{"source": route.from_node, "destination": route.to_node} for route in engine.disruption.affected_routes],
        "effect_types": [effect.type for effect in effects],
        "number_of_effects": len(effects),
        "duration_days": engine.disruption.duration,
        "severity": round(engine.disruption.severity, 3),
        "inventory_impact": magnitude({"inventory_loss"}),
        "transport_impact": magnitude({"route_closure", "shipping_delay"}),
        "demand_impact": magnitude({"demand_increase", "demand_drop"}),
        "capacity_impact": magnitude({"capacity_reduction", "throughput_reduction"}),
        "service_level": round(float(engine.nodes["retailer"].service_level), 3),
        "current_inventory": {key: round(float(node.inventory), 2) for key, node in engine.nodes.items()},
        "capacity_utilization": {key: round(float(node.inventory / max(node.capacity, 1)), 3) for key, node in engine.nodes.items()},
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
    disruption_type, target = _ardn_profile(disruption, getattr(engine, "physical_nodes", None))
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
        "type": disruption_type,
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
    # The trained ``service`` feature is an operational/customer-fulfilment
    # signal, not an inventory/capacity ratio.  Using stock fill here made a
    # healthy but buffered upstream node look disrupted to ARDN.  Preserve
    # the live engine's service signal for every tier; retailer service is the
    # same customer-fulfilment metric used by the simulator comparison.
    sim.service = np.clip(
        np.array([node.service_level for node in ordered_nodes], dtype=float),
        0.05,
        1.0,
    )
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
        "scenario_features": _scenario_features(engine),
    }


def _number(tensor: Any) -> float:
    return float(tensor.data.reshape(-1)[0])


def _cost_interval(cost: tuple[Any, Any, Any, Any], multiplier: float = 1.0) -> tuple[float, float]:
    mu, v, alpha, beta = (_number(item) for item in cost)
    variance = beta / (v * (alpha - 1.0) + 1e-6)
    sigma = max(variance, 1e-3) ** 0.5 * 20.0 * multiplier  # cost target was trained divided by 20
    point = mu * 20.0
    # Intervention costs cannot be negative; evidential intervals are
    # symmetric in model space, so project the lower endpoint to the valid
    # cost domain before exposing it to operators.
    return max(0.0, point - sigma), max(0.0, point + sigma)


def forecast_for_engine(engine: Any) -> dict[str, Any]:
    """Rank candidate actions and expose ARDN's forecast in JSON-safe values."""
    try:
        model, net, action_types = _load_ardn()
        forecasts = []
        for action_type in action_types:
            if action_type not in RUNTIME_TUNING["enabled_actions"]:
                continue
            episode = _episode_from_engine(engine, action_type, net)
            prediction = model.predict(episode)
            lower, upper = _cost_interval(prediction["cost"], RUNTIME_TUNING["cost_interval_multiplier"])
            recovery_variance = max(0.0, _number(prediction["T_rec_var"]))
            trajectory = [round(float(step.data.mean()), 3) for step in prediction["s_hat_seq"]]
            # Match the simulator's customer metric: maximum retailer
            # fulfilment drop from 100%, rather than the model's original
            # all-node deficit from the configurable 90% target.
            active_window = max(1, min(int(engine.disruption.duration), len(prediction["s_hat_seq"])))
            retailer_service_loss = max(
                0.0,
                1.0 - min(
                    float(step.data.reshape(-1)[-1])
                    for step in prediction["s_hat_seq"][:active_window]
                ),
            )
            risk_val = float(np.clip(_number(prediction["risk_p"]), 0.0, 1.0))
            ood_val = round(model.ood_score(episode), 2)
            ood_status = "In-Distribution (Typical)" if ood_val < 8.0 else ("Moderate Novelty" if ood_val <= 15.0 else "High Novelty (OOD Review Required)")
            risk_label = "<1% — Low severe-overflow risk" if risk_val < 0.01 else f"{risk_val:.1%} — Backlog overflow risk"

            # Feasibility evaluation under active network
            is_reroute = action_type == "reroute"
            imported = getattr(engine, "imported_network", None)
            has_alt_routes = (imported.get("source_node_count", 4) if isinstance(imported, dict) else 4) > 4
            feasibility = "Requires Multi-Route" if (is_reroute and not has_alt_routes) else "Feasible"

            forecasts.append({
                "action": action_type,
                "label": ACTION_LABELS[action_type],
                "recovery_days": round(max(0.0, _number(prediction["T_rec"])), 1),
                "recovery_definition": "ARDN expected recovery step for all predicted nodes reaching the configured service threshold",
                "recovery_uncertainty_days": round(recovery_variance ** 0.5, 1),
                "predicted_cost": round(_number(prediction["cost"][0]) * 20.0, 1),
                "cost_interval": [round(lower, 1), round(upper, 1)],
                "service_loss": round(retailer_service_loss, 3),
                "service_loss_definition": "max(1 - predicted retailer fulfilment) during active disruption window",
                "model_service_loss_functional": round(max(0.0, _number(prediction["L_service"])), 3),
                "risk_probability": round(risk_val, 3),
                "risk_label": risk_label,
                "trajectory": trajectory,
                "ood_score": ood_val,
                "ood_status": ood_status,
                "ood_threshold": 15.0,
                "feasibility": feasibility,
            })
        def scaled(item: dict[str, Any], key: str) -> float:
            values = [entry[key] for entry in forecasts]
            low, high = min(values), max(values)
            return 0.0 if high == low else (item[key] - low) / (high - low)

        for forecast in forecasts:
            forecast["ranking_score"] = round(
                RUNTIME_TUNING["recovery_weight"] * scaled(forecast, "recovery_days")
                + RUNTIME_TUNING["cost_weight"] * scaled(forecast, "predicted_cost")
                + RUNTIME_TUNING["risk_weight"] * scaled(forecast, "risk_probability")
                + RUNTIME_TUNING["service_loss_weight"] * scaled(forecast, "service_loss")
                + RUNTIME_TUNING["ood_weight"] * scaled(forecast, "ood_score"),
                3,
            )
            forecast["review_flags"] = [
                label for label, requires_review in {
                    "Risk above review threshold": forecast["risk_probability"] >= RUNTIME_TUNING["risk_review_threshold"],
                    "Novelty above review threshold": forecast["ood_score"] >= RUNTIME_TUNING["ood_review_threshold"],
                }.items() if requires_review
            ]
        forecasts.sort(key=lambda item: (item["ranking_score"], item["recovery_days"], item["predicted_cost"]))
        eligible = [item for item in forecasts if item["ood_score"] < RUNTIME_TUNING["ood_review_threshold"]]
        ood_review_required = not eligible
        # Novelty is a recommendation gate. If all candidates are novel we
        # retain the best advisory candidate, but explicitly block execution
        # pending review instead of silently continuing.
        recommended = min(eligible or forecasts, key=lambda item: (item["ranking_score"], item["recovery_days"], item["predicted_cost"]))
        recommended = {**recommended, "execution_status": "review_required" if (ood_review_required or recommended["review_flags"]) else "advisory_ready"}
        rec_summary = (
            f"Selected via multi-attribute decision criterion (ranking score {recommended['ranking_score']:.3f}). "
            f"Balances rapid recovery ({recommended['recovery_days']} ± {recommended['recovery_uncertainty_days']} days) "
            f"with low predicted service loss ({recommended['service_loss']:.3f}) and incremental cost (${recommended['predicted_cost']:.1f})."
        )
        return {
            "status": "ready",
            "model": "ARDN",
            "horizon_days": model.horizon,
            "runtime_tuning": RUNTIME_TUNING.copy(),
            "recommendation": {**recommended, "rationale": rec_summary},
            "alternatives": forecasts,
            "scenario_features": _scenario_features(engine),
            "note": "ARDN is an advisory neural predictor evaluated over a 15-day planning horizon. Predictions do not grant execution authority. The deterministic verifier remains the mandatory execution gate.",
            "comparison_contract": {
                "service_loss": "retailer demand-fulfilment drop from 100%, max during active disruption window",
                "simulator_peak_service_loss": "retailer demand-fulfilment drop from 100%, max during active disruption",
                "recovery": "ARDN expected model recovery step versus simulator elapsed day to service recovery or buffer stabilization",
                "ood_gate": "review_required when no candidate is below the configured novelty threshold",
            },
            "ood_review_required": ood_review_required,
        }
    except ARDNUnavailable as exc:
        return {"status": "unavailable", "model": "ARDN", "message": str(exc)}


def ardn_configuration() -> dict[str, Any]:
    """Expose the loaded model contract and active safe runtime controls."""
    try:
        model, _, _ = _load_ardn()
        return {
            "status": "ready",
            "model": "ARDN",
            "runtime_tuning": RUNTIME_TUNING.copy(),
            "trained_horizon": 15,
            "active_horizon": model.horizon,
            "architecture": {
                "node_feature_dimensions": 25,
                "edge_feature_dimensions": 5,
                "node_embedding_dimensions": 32,
                "graph_attention_layers": 2,
                "dynamics": "GRU rollout",
                "uncertainty": "Normal-Inverse-Gamma evidential heads",
                "ood": "Mahalanobis distance on graph embedding",
            },
            "captured_evaluation": {
                "held_out_trajectory_mse": 0.0058,
                "recovery_time_mae_steps": 2.17,
                "naive_recovery_time_mae_steps": 5.25,
                "mean_action_ranking_kendall_tau": 0.30,
                "full_ardn_ablation_mse": 0.0153,
                "no_gat_ablation_mse": 0.0077,
            },
            "note": "Runtime settings are applied to the active ARDN instance and new advisory forecasts. They do not overwrite trained_model.pkl.",
        }
    except ARDNUnavailable as exc:
        return {"status": "unavailable", "model": "ARDN", "message": str(exc)}


def update_ardn_runtime_tuning(tuning: Any) -> dict[str, Any]:
    """Apply bounded settings without changing model weights or the artifact."""
    model, _, _ = _load_ardn()
    RUNTIME_TUNING.update(tuning.model_dump())
    model.horizon = RUNTIME_TUNING["forecast_horizon"]
    model.tau = RUNTIME_TUNING["service_threshold"]
    return ardn_configuration()
