from .models import Agreement, Shipment
from .simulator import EDGES

def classical_policy(engine):
    shipments=[]
    for source,dest in [("supplier","manufacturer"),("manufacturer","distributor"),("distributor","retailer")]:
        a,b=engine.nodes[source],engine.nodes[dest]
        q=max(0,min(a.inventory,a.capacity,b.capacity-b.inventory,20))
        shipments.append(Shipment(**{"from":source,"to":dest,"quantity":q}))
    return Agreement(shipments=shipments)
