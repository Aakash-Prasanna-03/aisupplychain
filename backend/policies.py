from .models import Agreement, Shipment, Production
from .simulator import EDGES

def classical_policy(engine):
    shipments = []
    target_quantity = {"conservative": 15, "balanced": 18, "aggressive": 22}[engine.experiment.recovery_aggressiveness]
    for source, dest in [("supplier", "manufacturer"), ("manufacturer", "distributor"), ("distributor", "retailer")]:
        a, b = engine.nodes[source], engine.nodes[dest]
        # Proportional flow clamped to inventory and headroom
        q = max(0.0, min(a.inventory, a.capacity, b.capacity - b.inventory, target_quantity))
        shipments.append(Shipment(**{"from": source, "to": dest, "quantity": q}))
    production = [Production(node="manufacturer", quantity=target_quantity)]
    return Agreement(shipments=shipments, production=production)
