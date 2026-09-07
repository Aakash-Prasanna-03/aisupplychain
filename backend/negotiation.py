from .models import Agreement, Proposal
from .agents import agent_proposal
from .verifier import verify_agreement
from .policies import classical_policy

def build_agreement(proposals):
    chosen={}; production=[]
    for p in proposals:
        for s in p.shipments:
            key=(s.from_node,s.to); chosen[key]=max(chosen.get(key,0),s.quantity)
        production.extend(p.production)
    from .models import Shipment
    return Agreement(shipments=[Shipment(**{"from":a,"to":b,"quantity":q}) for (a,b),q in chosen.items()],production=production)

def negotiate(engine, verified=True):
    feedback=None; log=[]; attempts=0
    for round_no in range(1,5):
        proposals=[]
        for name in ["supplier","manufacturer","distributor","retailer"]:
            p, mocked=agent_proposal(name,engine,round_no,feedback); proposals.append(p)
            log.append({"speaker":f"{name.title()} Agent" + (" (Mock)" if mocked else ""),"message":p.reason,"proposal":p.model_dump(by_alias=True)})
        agreement=build_agreement(proposals); result=verify_agreement(engine,agreement); attempts+=1
        log.append({"speaker":"Verifier","message":"AGREEMENT VERIFIED" if result.valid else " · ".join(v["constraint"] for v in result.violations),"valid":result.valid,"violations":result.violations})
        if not verified or result.valid:
            return agreement,result,log,{"rounds":round_no,"attempts":attempts,"rejections":attempts-1,"status":"executed"}
        feedback=result.violations
    agreement=classical_policy(engine); result=verify_agreement(engine,agreement)
    log.append({"speaker":"System","message":"Negotiation exhausted; classical emergency policy applied."})
    return agreement,result,log,{"rounds":4,"attempts":attempts,"rejections":attempts,"status":"fallback"}
