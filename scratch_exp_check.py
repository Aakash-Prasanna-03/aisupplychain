from backend.models import SimulationRequest, Disruption, DisruptionEffect, ExperimentConfig
from backend.simulator import SimulationEngine
from backend.policies import classical_policy
from backend.negotiation import negotiate
from backend.verifier import verify_agreement
from backend.disruptions import effects, active

def compute_recovery_metrics(engine, disruption, horizon, tau=0.9):
    start = disruption.start_day
    history = engine.history
    levels = [h["service_level"] for h in history]
    avg_service = sum(levels) / len(levels) if levels else 1.0

    # Service during disruption window
    disr_levels = [h["service_level"] for h in history if active(disruption, h["day"])]
    min_service = min(disr_levels) if disr_levels else min(levels)
    peak_loss = max(0.0, 1.0 - min_service)

    # Recovery time calculation
    # Did service drop below tau?
    dropped = False
    recovery_day = None
    for h in history:
        if h["day"] >= start:
            if h["service_level"] < tau:
                dropped = True
            elif dropped and h["service_level"] >= tau:
                recovery_day = h["day"]
                break

    if dropped:
        rec_time = (recovery_day - start + 1) if recovery_day else (horizon - start + 1)
    else:
        # If service was buffered, calculate buffer stabilization time
        stab_day = None
        for h in history:
            if h["day"] >= start + disruption.duration:
                invs = h["inventories"]
                if all(invs.get(k, 0) >= 15.0 for k in ("supplier", "manufacturer", "distributor", "retailer")):
                    stab_day = h["day"]
                    break
        rec_time = (stab_day - start + 1) if stab_day else min(horizon, start + disruption.duration + 2)

    # Fairness variance: variance of downstream inventories / capacity
    final_nodes = engine.nodes
    downstream = [final_nodes[x].inventory / final_nodes[x].capacity for x in ["manufacturer", "distributor", "retailer"]]
    mean_d = sum(downstream) / len(downstream)
    fairness_var = sum((x - mean_d) ** 2 for x in downstream) / len(downstream)

    return {
        "recovery_time": round(rec_time, 1),
        "peak_service_level_loss": round(peak_loss, 3),
        "total_cost": round(engine.total_cost, 2),
        "average_service_level": round(avg_service, 3),
        "fairness_variance": round(fairness_var, 4),
    }

req = SimulationRequest(
    seed=42,
    disruption=Disruption(
        description="55% battery-cell shortage at supplier for 5 days",
        affected_nodes=["supplier"],
        effects=[DisruptionEffect(type="capacity_reduction", target="supplier", magnitude=0.55)],
        start_day=1,
        duration=5,
        severity=0.55
    ),
    experiment=ExperimentConfig(simulation_horizon=12, max_negotiation_rounds=4, recovery_aggressiveness="balanced")
)

print("Testing 3 modes...")
from backend.experiments import run_mode
for mode in ["classical", "unverified", "verified"]:
    res = run_mode(req, mode)
    print(f"Mode: {mode:12s} | Metrics: {res['metrics']}")
