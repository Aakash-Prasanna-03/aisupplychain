from .models import Agreement, VerificationResult
from .simulator import EDGES
from .disruptions import effects
from .config import MIN_SERVICE_LEVEL, FAIRNESS_THRESHOLD

def verify_agreement(engine, agreement: Agreement) -> VerificationResult:
    nodes=engine.nodes; violations=[]; balances={k:v.inventory for k,v in nodes.items()}; outgoing={k:0 for k in nodes}; fx=effects(engine.disruption, engine.day + 1)
    for s in agreement.shipments:
        if s.from_node not in nodes or s.to not in nodes or (s.from_node,s.to) not in EDGES:
            violations.append({"constraint":"INVALID_ROUTE","message":f"Shipment {s.from_node} → {s.to} is not a network route"}); continue
        if fx["route_closed"] or (s.from_node, s.to) in fx["closed_routes"]:
            violations.append({"constraint":"DISRUPTED_ROUTE","message":f"Route {s.from_node} → {s.to} is disrupted by the scenario"})
        outgoing[s.from_node]+=s.quantity; balances[s.from_node]-=s.quantity; balances[s.to]+=s.quantity
    for p in agreement.production:
        effective_limit=nodes[p.node].production_capacity * fx["capacity_factors"].get(p.node, 1) * fx["throughput_factors"].get(p.node, 1)
        if p.quantity > effective_limit:
            violations.append({"constraint":"PRODUCTION_LIMIT","message":f"{p.node} production limit is {effective_limit:.1f}"})
        balances[p.node]+=p.quantity
    for node,q in outgoing.items():
        if q > nodes[node].capacity:
            violations.append({"constraint":"CAPACITY_LIMIT","message":f"{node.title()} shipping capacity is {nodes[node].capacity}, requested {q}"})
        if q > nodes[node].inventory:
            violations.append({"constraint":"NEGATIVE_INVENTORY","message":f"{node.title()} has only {nodes[node].inventory} inventory, requested {q}"})
    for node,q in balances.items():
        if q < 0: violations.append({"constraint":"NEGATIVE_INVENTORY","message":f"{node.title()} would have negative inventory"})
        if q > nodes[node].capacity: violations.append({"constraint":"STORAGE_CAPACITY","message":f"{node.title()} would exceed storage capacity"})
    downstream=[balances[x]/nodes[x].capacity for x in ["manufacturer","distributor","retailer"]]
    mean=sum(downstream)/3; variance=sum((x-mean)**2 for x in downstream)/3
    projected=min(1,(balances["retailer"]+engine.last_demand*.5)/max(engine.last_demand,1))
    if projected < MIN_SERVICE_LEVEL: violations.append({"constraint":"MIN_SERVICE_LEVEL","message":f"Projected retailer service {projected:.0%} is below {MIN_SERVICE_LEVEL:.0%}"})
    if variance > FAIRNESS_THRESHOLD: violations.append({"constraint":"FAIRNESS","message":f"Downstream allocation variance {variance:.3f} exceeds limit"})
    return VerificationResult(valid=not violations, violations=violations, projected_metrics={"retailer_service_level":round(projected,3),"fairness_variance":round(variance,4)})
