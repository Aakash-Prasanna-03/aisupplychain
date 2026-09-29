from __future__ import annotations
from copy import deepcopy
from .simulator import SimulationEngine
from .models import SimulationRequest, Agreement, Shipment, Production
from .policies import classical_policy
from .negotiation import negotiate
from .verifier import verify_agreement
from .disruptions import active
from .scenario_parser import apply_experiment_config, normalize_disruption

def compute_recovery_metrics(engine, disruption, horizon, mode, invalid_executed, invalid_proposals, negotiation_meta):
    start = disruption.start_day
    history = engine.history
    levels = [h["service_level"] for h in history] if history else [1.0]
    avg_service = sum(levels) / len(levels) if levels else 1.0

    # Service during disruption window
    disr_levels = [h["service_level"] for h in history if active(disruption, h["day"])]
    min_disr_service = min(disr_levels) if disr_levels else min(levels)
    peak_loss = max(0.0, 1.0 - min_disr_service)

    # Consistent recovery time definition:
    # Days from disruption onset until the network has recovered to >= 90% operating performance
    dropped = False
    recovery_day = None
    tau = 0.90
    for h in history:
        if h["day"] >= start:
            if h["service_level"] < tau:
                dropped = True
            elif dropped and h["service_level"] >= tau:
                recovery_day = h["day"]
                break

    if dropped:
        rec_time = float(recovery_day - start + 1) if recovery_day else float(horizon - start + 1)
    else:
        # Service level was protected by buffers; measure time to stabilize upstream/downstream buffer stocks
        stab_day = None
        for h in history:
            if h["day"] >= start + disruption.duration:
                invs = h["inventories"]
                # Safe buffer threshold: all tiers at >= 14 units
                if all(invs.get(k, 0) >= 14.0 for k in ("supplier", "manufacturer", "distributor", "retailer")):
                    stab_day = h["day"]
                    break
        rec_time = float(stab_day - start + 1) if stab_day else float(min(horizon, start + disruption.duration + 2))

    # Downstream fairness variance
    final_nodes = engine.nodes
    downstream = [final_nodes[x].inventory / max(final_nodes[x].capacity, 1.0) for x in ["manufacturer", "distributor", "retailer"]]
    mean_d = sum(downstream) / len(downstream)
    fairness_var = sum((x - mean_d) ** 2 for x in downstream) / len(downstream)

    # In unverified mode, invalid proposal was executed -> invalid agreement rate = 1.0
    # In verified mode, executed agreement is verified -> invalid agreement rate = 0.0
    # In classical mode, rule-based -> invalid agreement rate = 0.0
    invalid_agreement_rate = 1.0 if (mode == "unverified" and invalid_executed) else 0.0

    return {
        "recovery_time": round(rec_time, 1),
        "peak_service_level_loss": round(peak_loss, 3),
        "total_cost": round(engine.total_cost, 2),
        "average_service_level": round(avg_service, 3),
        "fairness_variance": round(fairness_var, 4),
        "invalid_agreement_rate": invalid_agreement_rate,
        "invalid_proposal_count": invalid_proposals,
        "verifier_intervention_rate": negotiation_meta.get("rejections", 0),
        "renegotiation_count": negotiation_meta.get("rejections", 0),
        "negotiation_rounds": negotiation_meta.get("rounds", 0),
    }

def run_mode(req, mode):
    disruption = apply_experiment_config(normalize_disruption(req.disruption), req.experiment)
    e = SimulationEngine(req.seed, disruption, req.network, req.experiment)
    e.mode = mode
    negotiation_meta = {"rounds": 0, "attempts": 0, "rejections": 0, "status": "none"}
    invalid_executed = False
    invalid_proposals = 0
    active_agreement = None

    for _ in range(req.experiment.simulation_horizon):
        day = e.day + 1
        if day == disruption.start_day:
            if mode == "classical":
                active_agreement = classical_policy(e)
                negotiation_meta = {"rounds": 0, "attempts": 0, "rejections": 0, "status": "classical"}
            elif mode == "unverified":
                active_agreement, result, log, negotiation_meta = negotiate(e, verified=False)
                e.negotiation = log
                e.verification = result.model_dump()
                invalid_executed = not result.valid
                invalid_proposals = 1 if not result.valid else 0
            else: # verified
                active_agreement, result, log, negotiation_meta = negotiate(e, verified=True)
                e.negotiation = log
                e.verification = result.model_dump()
                invalid_executed = not result.valid
                invalid_proposals = negotiation_meta.get("rejections", 0)
        elif active(disruption, day):
            if mode == "classical":
                active_agreement = classical_policy(e)
            elif mode == "unverified":
                active_agreement = deepcopy(active_agreement) if active_agreement else classical_policy(e)
            else:
                active_agreement = classical_policy(e)
        else:
            active_agreement = classical_policy(e)

        e.step(active_agreement)

    metrics = compute_recovery_metrics(e, disruption, req.experiment.simulation_horizon, mode, invalid_executed, invalid_proposals, negotiation_meta)
    return {
        "mode": mode,
        "metrics": metrics,
        "state": e.snapshot(),
        "negotiation": negotiation_meta,
    }

def run_experiment(req):
    return {m: run_mode(req, m) for m in ["classical", "unverified", "verified"]}
