from .models import Agreement, Shipment
from .simulator import EDGES

def classical_policy(engine):
    shipments=[]
    target_quantity={"conservative":15,"balanced":20,"aggressive":25}[engine.experiment.recovery_aggressiveness]
    for source,dest in [("supplier","manufacturer"),("manufacturer","distributor"),("distributor","retailer")]:
        a,b=engine.nodes[source],engine.nodes[dest]
        q=max(0,min(a.inventory,a.capacity,b.capacity-b.inventory,target_quantity))
        shipments.append(Shipment(**{"from":source,"to":dest,"quantity":q}))
    return Agreement(shipments=shipments)
