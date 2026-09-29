import random
import numpy as np
from backend.models import Node, Disruption, DisruptionEffect, Agreement, Shipment, Production, SimulationRequest, ExperimentConfig
from backend.verifier import verify_agreement

def create_nodes():
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

print("Starting prototype comparison test...")
