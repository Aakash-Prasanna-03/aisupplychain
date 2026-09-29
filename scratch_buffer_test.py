import numpy as np
from backend.models import Node, Disruption, DisruptionEffect
from backend.disruptions import effects

def test_inv(init_inv=16):
    nodes = {
        "supplier": Node(id="supplier", name="Supplier", inventory=init_inv, capacity=40, production_capacity=22, holding_cost=0.2, shortage_cost=4.0),
        "manufacturer": Node(id="manufacturer", name="Manufacturer", inventory=init_inv, capacity=40, production_capacity=22, holding_cost=0.2, shortage_cost=4.0),
        "distributor": Node(id="distributor", name="Distributor", inventory=init_inv, capacity=40, production_capacity=0, holding_cost=0.2, shortage_cost=4.0),
        "retailer": Node(id="retailer", name="Retailer", inventory=init_inv, capacity=40, production_capacity=0, holding_cost=0.2, shortage_cost=4.0),
    }

    disruption = Disruption(
        description="55% battery-cell shortage at supplier for 5 days",
        affected_nodes=["supplier"],
        effects=[DisruptionEffect(type="capacity_reduction", target="supplier", magnitude=0.55)],
        start_day=1,
        duration=5,
        severity=0.55
    )

    rng = np.random.default_rng(42)
    history = []
    total_cost = 0.0

    for day in range(1, 13):
        fx = effects(disruption, day)
        sup = nodes["supplier"]
        sup_factor = fx["capacity_factor"] * fx["capacity_factors"].get("supplier", 1.0)
        sup_prod = min(sup.production_capacity * sup_factor, sup.capacity - sup.inventory)
        sup.inventory += sup_prod

        # Classical: each tier tries to ship 20
        q_sm = min(20, sup.inventory, nodes["manufacturer"].capacity - nodes["manufacturer"].inventory)
        sup.inventory -= q_sm
        nodes["manufacturer"].inventory += q_sm

        q_md = min(20, nodes["manufacturer"].inventory, nodes["distributor"].capacity - nodes["distributor"].inventory)
        nodes["manufacturer"].inventory -= q_md
        nodes["distributor"].inventory += q_md

        q_dr = min(20, nodes["distributor"].inventory, nodes["retailer"].capacity - nodes["retailer"].inventory)
        nodes["distributor"].inventory -= q_dr
        nodes["retailer"].inventory += q_dr

        demand = round(20 * fx["demand_factor"] + rng.integers(-2, 3), 1)
        ret = nodes["retailer"]
        served = min(ret.inventory, demand)
        ret.inventory -= served
        service_level = served / demand if demand else 1.0

        for n in nodes.values():
            total_cost += n.inventory * n.holding_cost
        total_cost += (demand - served) * ret.shortage_cost

        history.append({"day": day, "demand": demand, "served": served, "service_level": service_level, "retailer_inv": ret.inventory})

    print(f"Initial Inv = {init_inv}:")
    print("Service levels:", [round(h["service_level"], 2) for h in history])
    print("Retailer invs:", [round(h["retailer_inv"], 1) for h in history])
    print(f"Total cost: ${total_cost:.1f}")

test_inv(16)
test_inv(14)
test_inv(12)
