import random
from backend.models import Node, Disruption, DisruptionEffect, Agreement, Shipment, Production, SimulationRequest, ExperimentConfig
from backend.verifier import verify_agreement
from backend.policies import classical_policy

# Let's inspect the 3 modes under realistic inventory parameters
def create_nodes():
    return {
        "supplier": Node(id="supplier", name="Supplier", inventory=30, capacity=60, production_capacity=25),
        "manufacturer": Node(id="manufacturer", name="Manufacturer", inventory=25, capacity=50, production_capacity=25),
        "distributor": Node(id="distributor", name="Distributor", inventory=20, capacity=40, production_capacity=0),
        "retailer": Node(id="retailer", name="Retailer", inventory=20, capacity=35, production_capacity=0),
    }

disruption = Disruption(
    description="55% battery-cell shortage at supplier for 5 days",
    affected_nodes=["supplier"],
    effects=[DisruptionEffect(type="capacity_reduction", target="supplier", magnitude=0.55)],
    start_day=1,
    duration=5,
    severity=0.55
)

# Test classical policy under a 55% reduction
print("Checking classical policy on day 1:")
from backend.simulator import SimulationEngine
e = SimulationEngine(seed=42, disruption=disruption)
e.nodes = create_nodes()
v = verify_agreement(e, classical_policy(e))
print("Classical policy verification:", v.valid, v.violations)
