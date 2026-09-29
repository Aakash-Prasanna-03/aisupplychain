from __future__ import annotations
import random
from copy import deepcopy
from .models import Node, Agreement, Disruption, ImportedNode, ExperimentConfig
from .disruptions import effects, active
from .config import LLM_CONFIGURED, LLM_MODEL

ORDER = ["supplier", "manufacturer", "distributor", "retailer"]
EDGES = {("supplier", "manufacturer"), ("manufacturer", "distributor"), ("distributor", "retailer")}

def default_nodes(imported_nodes: list[ImportedNode] | None = None):
    # Nominal parameters: steady-state demand is ~20 units/day.
    # Production capacity is 25 units/day with 1-1.5 days safety buffer.
    defaults = [
        Node(id="supplier", name="Supplier", inventory=28, capacity=50, production_capacity=25, holding_cost=0.2, shortage_cost=4.0, service_level_target=0.9),
        Node(id="manufacturer", name="Manufacturer", inventory=22, capacity=45, production_capacity=25, holding_cost=0.2, shortage_cost=4.0, service_level_target=0.9),
        Node(id="distributor", name="Distributor", inventory=18, capacity=35, production_capacity=0, holding_cost=0.2, shortage_cost=4.0, service_level_target=0.9),
        Node(id="retailer", name="Retailer", inventory=18, capacity=30, production_capacity=0, holding_cost=0.2, shortage_cost=4.0, service_level_target=0.9),
    ]
    if not imported_nodes:
        return {n.id: n for n in defaults}
    grouped = {tier: [node for node in imported_nodes if node.operating_tier == tier] for tier in ORDER}
    nodes = []
    for fallback in defaults:
        group = grouped[fallback.id]
        count = len(group)
        nodes.append(Node(
            id=fallback.id,
            name=group[0].name if count == 1 else f"{fallback.id.title()} network ({count} nodes)",
            inventory=sum(node.inventory for node in group),
            capacity=sum(node.capacity for node in group),
            production_capacity=sum(node.production_capacity for node in group),
            holding_cost=sum(node.holding_cost for node in group) / count,
            shortage_cost=sum(node.shortage_cost for node in group) / count,
            service_level_target=sum(node.service_level_target for node in group) / count,
            status="NORMAL",
            status_detail="Operating within nominal limits",
        ))
    return {node.id: node for node in nodes}

class SimulationEngine:
    def __init__(self, seed=42, disruption=None, network=None, experiment=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes = default_nodes(network.nodes if network else None)
        self.day = 0
        self.imported_network = {
            "active": bool(network),
            "source_node_count": len(network.nodes) if network else 4,
            "aggregation": "tier totals" if network else "default network"
        }
        self.disruption = disruption or Disruption()
        self.history = []
        self.total_cost = 0.0
        self.demands = []
        self.experiment = experiment or ExperimentConfig()
        self.last_demand = 20.0
        self.negotiation = []
        self.verification = None
        self.mode = "verified"
        self.mock_agents = not LLM_CONFIGURED
        self.last_agreement = None

    def snapshot(self):
        is_active = active(self.disruption, self.day)
        # Network operational status classification
        if is_active:
            net_status = f"Active Disruption (Day {self.day} of {self.disruption.duration})"
        elif any(n.status in ("DISRUPTED", "CONSTRAINED", "BUFFERING", "AT RISK") for n in self.nodes.values()):
            net_status = "Recovering (Stabilizing buffer stocks)"
        else:
            net_status = "Recovered (Operating within targets)"

        levels = [h["service_level"] for h in self.history] if self.history else [1.0]
        avg_service = sum(levels) / len(levels) if levels else 1.0
        disr_levels = [h["service_level"] for h in self.history if active(self.disruption, h["day"])]
        disr_service = sum(disr_levels) / len(disr_levels) if disr_levels else avg_service

        return {
            "day": self.day,
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "total_cost": round(self.total_cost, 2),
            "history": self.history,
            "negotiation": self.negotiation,
            "verification": self.verification,
            "mode": self.mode,
            "mock_agents": self.mock_agents,
            "agent_model": "mock" if self.mock_agents else LLM_MODEL,
            "imported_network": self.imported_network,
            "scenario": self.disruption.model_dump(),
            "experiment": self.experiment.model_dump(),
            "network_status": net_status,
            "is_disruption_active": is_active,
            "disruption_service": round(disr_service, 3),
            "average_service": round(avg_service, 3),
        }

    def step(self, agreement: Agreement | None = None):
        self.day += 1
        fx = effects(self.disruption, self.day)
        is_active = active(self.disruption, self.day)
        self.last_agreement = agreement

        # 1. Apply sudden inventory loss shocks on start day
        for node_id, loss in fx["inventory_losses"].items():
            if node_id in self.nodes:
                loss_qty = self.nodes[node_id].inventory * min(loss, 1.0)
                self.nodes[node_id].inventory = max(0.0, self.nodes[node_id].inventory - loss_qty)

        # 2. Upstream supplier production
        supplier = self.nodes["supplier"]
        supplier_factor = fx["capacity_factor"] * fx["capacity_factors"].get("supplier", 1.0) * fx["throughput_factors"].get("supplier", 1.0)
        prod_limit = supplier.production_capacity * supplier_factor
        supplier.inventory = min(supplier.capacity, supplier.inventory + min(prod_limit, max(0.0, supplier.capacity - supplier.inventory)))

        # 3. Execute negotiated agreement or baseline policy
        if agreement:
            # Production orders (e.g. manufacturer)
            for p in agreement.production:
                if p.node in self.nodes:
                    node = self.nodes[p.node]
                    factor = fx["capacity_factors"].get(p.node, 1.0) * fx["throughput_factors"].get(p.node, 1.0)
                    eff_prod = min(p.quantity, node.production_capacity * factor)
                    node.inventory = min(node.capacity, node.inventory + eff_prod)

            # Shipments
            for s in agreement.shipments:
                route_closed = fx["route_closed"] or (s.from_node, s.to) in fx["closed_routes"]
                if (s.from_node, s.to) in EDGES:
                    if route_closed:
                        # In unverified mode, shipping on a closed route loses goods
                        if self.mode == "unverified":
                            lost = min(s.quantity, self.nodes[s.from_node].inventory)
                            self.nodes[s.from_node].inventory -= lost
                            self.total_cost += lost * 2.0  # transport disruption loss penalty
                        continue

                    # Execute flow
                    if self.mode == "unverified":
                        # Unverified mode allows overflowing destination capacity (causes warehouse congestion)
                        q = min(s.quantity, self.nodes[s.from_node].inventory)
                        self.nodes[s.from_node].inventory -= q
                        self.nodes[s.to].inventory += q
                    else:
                        # Verified mode strictly respects storage capacities
                        q = min(s.quantity, self.nodes[s.from_node].inventory, max(0.0, self.nodes[s.to].capacity - self.nodes[s.to].inventory))
                        self.nodes[s.from_node].inventory -= q
                        self.nodes[s.to].inventory += q

        # 4. Customer demand arrival at retailer
        demand = round(20 * fx["demand_factor"] + self.rng.randint(-2, 2), 1)
        self.demands.append(demand)
        self.last_demand = demand

        retailer = self.nodes["retailer"]
        served = min(retailer.inventory, demand)
        retailer.inventory -= served
        retailer.service_level = round(served / demand if demand else 1.0, 3)

        # 5. Compute holding, shortage, and congestion costs
        for node in self.nodes.values():
            self.total_cost += node.inventory * node.holding_cost
            # If node inventory exceeded capacity (unverified mode overflow)
            if node.inventory > node.capacity:
                overflow = node.inventory - node.capacity
                self.total_cost += overflow * 1.5  # warehouse overflow/congestion penalty

        self.total_cost += (demand - served) * retailer.shortage_cost

        # 6. Update realistic operational status for each node
        # Supplier status
        if is_active and supplier_factor < 0.95:
            supplier.status = "DISRUPTED"
            supplier.status_detail = f"Production constrained to {supplier_factor:.0%} of normal capacity"
            supplier.fulfillment_rate = round(supplier_factor, 2)
        elif supplier.inventory < 15:
            supplier.status = "RECOVERING"
            supplier.status_detail = f"Replenishing buffer inventory ({supplier.inventory:.0f}/{supplier.capacity:.0f})"
            supplier.fulfillment_rate = 1.0
        else:
            supplier.status = "NORMAL"
            supplier.status_detail = "Operating at full nominal capacity"
            supplier.fulfillment_rate = 1.0

        # Manufacturer status
        manufacturer = self.nodes["manufacturer"]
        if is_active and supplier_factor < 0.95:
            if manufacturer.inventory > 10:
                manufacturer.status = "BUFFERING"
                manufacturer.status_detail = f"Operating from buffer stock ({manufacturer.inventory:.0f}/{manufacturer.capacity:.0f} units)"
            else:
                manufacturer.status = "CONSTRAINED"
                manufacturer.status_detail = f"Inbound supply constrained by upstream shortage ({manufacturer.inventory:.0f} units)"
        elif manufacturer.inventory < 12:
            manufacturer.status = "RECOVERING"
            manufacturer.status_detail = f"Rebuilding stock buffer ({manufacturer.inventory:.0f}/{manufacturer.capacity:.0f})"
        else:
            manufacturer.status = "NORMAL"
            manufacturer.status_detail = "Production and inventory balanced"

        # Distributor status
        distributor = self.nodes["distributor"]
        if is_active and (supplier_factor < 0.95 or manufacturer.inventory < 12):
            if distributor.inventory > 10:
                distributor.status = "BUFFERING"
                distributor.status_detail = f"Drawing on buffer stock ({distributor.inventory:.0f}/{distributor.capacity:.0f} units)"
            else:
                distributor.status = "CONSTRAINED"
                distributor.status_detail = f"Low allocation received from upstream ({distributor.inventory:.0f} units)"
        elif distributor.inventory < 10:
            distributor.status = "RECOVERING"
            distributor.status_detail = f"Replenishing distribution buffer ({distributor.inventory:.0f}/{distributor.capacity:.0f})"
        else:
            distributor.status = "NORMAL"
            distributor.status_detail = "Normal distribution flow"

        # Retailer status
        if retailer.service_level < 0.7:
            retailer.status = "AT RISK"
            retailer.status_detail = f"Customer stockout! Service degraded to {retailer.service_level:.0%}"
        elif retailer.service_level < 0.95:
            retailer.status = "CONSTRAINED"
            retailer.status_detail = f"Service constrained at {retailer.service_level:.0%}"
        else:
            if is_active:
                retailer.status = "NORMAL"
                retailer.status_detail = f"100% fulfillment maintained via inventory buffer ({retailer.inventory:.0f} remaining)"
            elif any(h["service_level"] < 0.95 for h in self.history):
                retailer.status = "RECOVERED"
                retailer.status_detail = "Recovered: Customer demand fully fulfilled"
            else:
                retailer.status = "NORMAL"
                retailer.status_detail = "Operating normally within target"

        # Record history step
        self.history.append({
            "day": self.day,
            "demand": demand,
            "served": served,
            "service_level": retailer.service_level,
            "disruption_active": is_active,
            "inventories": {k: round(v.inventory, 1) for k, v in self.nodes.items()},
            "node_statuses": {k: v.status for k, v in self.nodes.items()},
            "total_cost": round(self.total_cost, 2),
        })

        return self.snapshot()
