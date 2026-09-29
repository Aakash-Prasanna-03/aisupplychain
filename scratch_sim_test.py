import numpy as np
from backend.models import Node, Disruption, DisruptionEffect, ExperimentConfig, Agreement, Shipment
from backend.simulator import SimulationEngine, default_nodes
from backend.policies import classical_policy
from backend.disruptions import effects

# Let's inspect default_nodes and a 55% shortage at supplier
disruption = Disruption(
    description="55% battery-cell shortage at supplier for 5 days",
    affected_nodes=["supplier"],
    effects=[DisruptionEffect(type="capacity_reduction", target="supplier", magnitude=0.55)],
    start_day=1,
    duration=5,
    severity=0.55
)

# Test 1: with current defaults
e1 = SimulationEngine(seed=42, disruption=disruption)
print("Initial e1 nodes:", {k: (v.inventory, v.capacity, v.production_capacity) for k, v in e1.nodes.items()})
for day in range(12):
    e1.step(classical_policy(e1))
print("Final e1 retailer service:", [h["service_level"] for h in e1.history])

# Test 2: with calibrated realistic parameters
# Where normal demand is ~20, normal capacity is 25
def calibrated_nodes():
    return {
        "supplier": Node(id="supplier", name="Supplier", inventory=35, capacity=60, production_capacity=25),
        "manufacturer": Node(id="manufacturer", name="Manufacturer", inventory=30, capacity=50, production_capacity=25),
        "distributor": Node(id="distributor", name="Distributor", inventory=25, capacity=40, production_capacity=0),
        "retailer": Node(id="retailer", name="Retailer", inventory=20, capacity=35, production_capacity=0),
    }

e2 = SimulationEngine(seed=42, disruption=disruption)
e2.nodes = calibrated_nodes()
for day in range(12):
    e2.step(classical_policy(e2))
print("Final e2 retailer service (classical):", [round(h["service_level"], 2) for h in e2.history])
print("Final e2 inventories:", {k: round(v.inventory, 1) for k, v in e2.nodes.items()})
