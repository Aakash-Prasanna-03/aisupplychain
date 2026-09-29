import numpy as np
from backend.models import Node, Disruption, DisruptionEffect, Agreement, Shipment, Production, ExperimentConfig
from backend.verifier import verify_agreement
from backend.disruptions import effects

def create_initial_nodes():
    return {
        "supplier": Node(id="supplier", name="Supplier", inventory=28, capacity=50, production_capacity=25, holding_cost=0.2, shortage_cost=4.0),
        "manufacturer": Node(id="manufacturer", name="Manufacturer", inventory=22, capacity=45, production_capacity=25, holding_cost=0.2, shortage_cost=4.0),
        "distributor": Node(id="distributor", name="Distributor", inventory=18, capacity=35, production_capacity=0, holding_cost=0.2, shortage_cost=4.0),
        "retailer": Node(id="retailer", name="Retailer", inventory=18, capacity=30, production_capacity=0, holding_cost=0.2, shortage_cost=4.0),
    }

disruption = Disruption(
    description="55% battery-cell shortage at supplier for 5 days",
    affected_nodes=["supplier"],
    effects=[DisruptionEffect(type="capacity_reduction", target="supplier", magnitude=0.55)],
    start_day=1,
    duration=5,
    severity=0.55
)

# Simulate classical policy for 12 days
nodes = create_initial_nodes()
history_classical = []
rng = np.random.default_rng(42)
total_cost_classical = 0.0

for day in range(1, 13):
    fx = effects(disruption, day)
    sup = nodes["supplier"]
    # Supplier production with disruption factor
    sup_factor = fx["capacity_factor"] * fx["capacity_factors"].get("supplier", 1.0)
    sup_prod = min(sup.production_capacity * sup_factor, sup.capacity - sup.inventory)
    sup.inventory += sup_prod

    # Classical policy shipments: each tier orders 20
    # Supplier to Manufacturer
    q_sm = min(20, sup.inventory, nodes["manufacturer"].capacity - nodes["manufacturer"].inventory)
    sup.inventory -= q_sm
    nodes["manufacturer"].inventory += q_sm

    # Manufacturer to Distributor
    q_md = min(20, nodes["manufacturer"].inventory, nodes["distributor"].capacity - nodes["distributor"].inventory)
    nodes["manufacturer"].inventory -= q_md
    nodes["distributor"].inventory += q_md

    # Distributor to Retailer
    q_dr = min(20, nodes["distributor"].inventory, nodes["retailer"].capacity - nodes["retailer"].inventory)
    nodes["distributor"].inventory -= q_dr
    nodes["retailer"].inventory += q_dr

    # Demand
    demand = round(20 * fx["demand_factor"] + rng.integers(-2, 3), 1)
    ret = nodes["retailer"]
    served = min(ret.inventory, demand)
    ret.inventory -= served
    service_level = served / demand if demand else 1.0

    # Costs
    for n in nodes.values():
        total_cost_classical += n.inventory * n.holding_cost
    total_cost_classical += (demand - served) * ret.shortage_cost

    history_classical.append({
        "day": day,
        "demand": demand,
        "served": served,
        "service_level": service_level,
        "inventories": {k: round(v.inventory, 1) for k, v in nodes.items()}
    })

print("Classical Service Levels:")
print([round(h["service_level"], 2) for h in history_classical])
print("Classical Inventories:")
for h in history_classical:
    print(f"Day {h['day']:2d}: Serv={h['service_level']:.2f}, Invs={h['inventories']}")
print(f"Total Cost: ${total_cost_classical:.1f}")
