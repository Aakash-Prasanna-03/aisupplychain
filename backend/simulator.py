from __future__ import annotations
import random
from copy import deepcopy
from .models import Node, Agreement, Disruption, ImportedNode, ExperimentConfig
from .disruptions import effects
from .config import LLM_CONFIGURED, LLM_MODEL

ORDER = ["supplier", "manufacturer", "distributor", "retailer"]
EDGES = {("supplier","manufacturer"), ("manufacturer","distributor"), ("distributor","retailer")}

def default_nodes(imported_nodes: list[ImportedNode] | None = None):
    defaults = [
        Node(id="supplier",name="Supplier",inventory=100,capacity=80,production_capacity=80),
        Node(id="manufacturer",name="Manufacturer",inventory=60,capacity=60,production_capacity=50),
        Node(id="distributor",name="Distributor",inventory=40,capacity=50,production_capacity=0),
        Node(id="retailer",name="Retailer",inventory=30,capacity=40,production_capacity=0),
    ]
    if not imported_nodes:
        return {n.id:n for n in defaults}
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
        ))
    return {node.id: node for node in nodes}

class SimulationEngine:
    def __init__(self, seed=42, disruption=None, network=None, experiment=None):
        self.seed=seed; self.rng=random.Random(seed); self.nodes=default_nodes(network.nodes if network else None); self.day=0
        self.imported_network={"active":bool(network),"source_node_count":len(network.nodes) if network else 4,"aggregation":"tier totals" if network else "default network"}
        self.disruption=disruption or Disruption(); self.history=[]; self.total_cost=0.; self.demands=[]
        self.experiment=experiment or ExperimentConfig()
        self.last_demand=20.; self.negotiation=[]; self.verification=None; self.mode="verified"; self.mock_agents=not LLM_CONFIGURED
    def snapshot(self):
        return {"day":self.day,"nodes":[n.model_dump() for n in self.nodes.values()],"total_cost":round(self.total_cost,2),"history":self.history,"negotiation":self.negotiation,"verification":self.verification,"mode":self.mode,"mock_agents":self.mock_agents,"agent_model":"mock" if self.mock_agents else LLM_MODEL,"imported_network":self.imported_network,"scenario":self.disruption.model_dump(),"experiment":self.experiment.model_dump()}
    def step(self, agreement: Agreement|None=None):
        self.day += 1; fx=effects(self.disruption,self.day)
        for node_id, loss in fx["inventory_losses"].items():
            self.nodes[node_id].inventory = max(0, self.nodes[node_id].inventory * (1 - min(loss, 1)))
        supplier=self.nodes["supplier"]
        supplier_factor=fx["capacity_factor"] * fx["capacity_factors"].get("supplier", 1) * fx["throughput_factors"].get("supplier", 1)
        supplier.inventory=min(supplier.capacity, supplier.inventory + min(supplier.production_capacity*supplier_factor, supplier.capacity-supplier.inventory))
        if agreement:
            for p in agreement.production:
                node=self.nodes[p.node]; factor=fx["capacity_factors"].get(p.node, 1) * fx["throughput_factors"].get(p.node, 1); node.inventory=min(node.capacity,node.inventory+min(p.quantity,node.production_capacity*factor))
            for s in agreement.shipments:
                route_closed=fx["route_closed"] or (s.from_node,s.to) in fx["closed_routes"]
                if (s.from_node,s.to) in EDGES and not route_closed:
                    q=min(s.quantity,self.nodes[s.from_node].inventory,self.nodes[s.to].capacity-self.nodes[s.to].inventory)
                    self.nodes[s.from_node].inventory-=q; self.nodes[s.to].inventory+=q
        demand=round(20*fx["demand_factor"] + self.rng.randint(-2,2),1); self.demands.append(demand); self.last_demand=demand
        retailer=self.nodes["retailer"]; served=min(retailer.inventory,demand); retailer.inventory-=served
        retailer.service_level=served/demand if demand else 1; retailer.status="AT RISK" if retailer.service_level<.7 else "NORMAL"
        for node in self.nodes.values():
            self.total_cost += node.inventory*node.holding_cost
        self.total_cost += (demand-served)*retailer.shortage_cost
        self.history.append({"day":self.day,"demand":demand,"served":served,"service_level":retailer.service_level,"inventories":{k:round(v.inventory,1) for k,v in self.nodes.items()}})
        return self.snapshot()
