from __future__ import annotations
import random
from copy import deepcopy
from .models import Node, Agreement, Disruption
from .disruptions import effects
from .config import LLM_CONFIGURED, LLM_MODEL

ORDER = ["supplier", "manufacturer", "distributor", "retailer"]
EDGES = {("supplier","manufacturer"), ("manufacturer","distributor"), ("distributor","retailer")}

def default_nodes():
    return {n.id:n for n in [
        Node(id="supplier",name="Supplier",inventory=100,capacity=80,production_capacity=80),
        Node(id="manufacturer",name="Manufacturer",inventory=60,capacity=60,production_capacity=50),
        Node(id="distributor",name="Distributor",inventory=40,capacity=50,production_capacity=0),
        Node(id="retailer",name="Retailer",inventory=30,capacity=40,production_capacity=0),
    ]}

class SimulationEngine:
    def __init__(self, seed=42, disruption=None):
        self.seed=seed; self.rng=random.Random(seed); self.nodes=default_nodes(); self.day=0
        self.disruption=disruption or Disruption(); self.history=[]; self.total_cost=0.; self.demands=[]
        self.last_demand=20.; self.negotiation=[]; self.verification=None; self.mode="verified"; self.mock_agents=not LLM_CONFIGURED
    def snapshot(self):
        return {"day":self.day,"nodes":[n.model_dump() for n in self.nodes.values()],"total_cost":round(self.total_cost,2),"history":self.history,"negotiation":self.negotiation,"verification":self.verification,"mode":self.mode,"mock_agents":self.mock_agents,"agent_model":"mock" if self.mock_agents else LLM_MODEL}
    def step(self, agreement: Agreement|None=None):
        self.day += 1; fx=effects(self.disruption,self.day)
        supplier=self.nodes["supplier"]
        supplier.inventory=min(supplier.capacity, supplier.inventory + min(supplier.production_capacity*fx["capacity_factor"], supplier.capacity-supplier.inventory))
        if agreement:
            for p in agreement.production:
                node=self.nodes[p.node]; node.inventory=min(node.capacity,node.inventory+p.quantity)
            for s in agreement.shipments:
                if (s.from_node,s.to) in EDGES and not (fx["route_closed"] and s.from_node=="supplier"):
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
