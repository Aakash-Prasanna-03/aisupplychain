from backend.models import Disruption, DisruptionEffect, ExperimentConfig
from backend.simulator import SimulationEngine
from backend.ardn_service import forecast_for_engine

disruption = Disruption(
    description="55% battery-cell shortage at supplier for 5 days",
    affected_nodes=["supplier"],
    effects=[DisruptionEffect(type="capacity_reduction", target="supplier", magnitude=0.55)],
    start_day=1,
    duration=5,
    severity=0.55
)
e = SimulationEngine(seed=42, disruption=disruption)
fc = forecast_for_engine(e)
print("ARDN status:", fc["status"])
print("Recommended:", fc.get("recommendation", {}).get("action"), fc.get("recommendation", {}).get("label"))
print("\nAll 6 options:")
for opt in fc.get("alternatives", []):
    print(f"Action: {opt['action']:16s} | Label: {opt['label']:25s} | Rec: {opt['recovery_days']:4.1f} ± {opt['recovery_uncertainty_days']:3.1f} d | Cost: ${opt['predicted_cost']:5.1f} [{opt['cost_interval'][0]:.1f}, {opt['cost_interval'][1]:.1f}] | ServLoss: {opt['service_loss']:.3f} | Risk: {opt['risk_probability']:.3f} | OOD: {opt['ood_score']:5.2f} | Score: {opt['ranking_score']:.3f}")
