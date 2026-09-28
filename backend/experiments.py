from copy import deepcopy
from .simulator import SimulationEngine
from .models import SimulationRequest
from .policies import classical_policy
from .negotiation import negotiate
from .verifier import verify_agreement

def run_mode(req, mode):
    e=SimulationEngine(req.seed,req.disruption,req.network); e.mode=mode; negotiation={"rounds":0,"attempts":0,"rejections":0,"status":"none"}; invalid=False
    for _ in range(req.simulation_days):
        agreement=classical_policy(e)
        if e.day+1==req.disruption.start_day:
            if mode=="classical": negotiation={"rounds":0,"attempts":0,"rejections":0,"status":"classical"}
            else:
                agreement,result,log,negotiation=negotiate(e,mode=="verified"); e.negotiation=log; e.verification=result.model_dump(); invalid=not result.valid
        e.step(agreement)
    levels=[h["service_level"] for h in e.history]; loss=1-min(levels); recovery=next((h["day"]-req.disruption.start_day for h in e.history if h["day"]>=req.disruption.start_day and h["service_level"]>=.9),req.simulation_days-req.disruption.start_day+1)
    return {"mode":mode,"metrics":{"recovery_time":recovery,"peak_service_level_loss":round(loss,3),"total_cost":round(e.total_cost,2),"average_service_level":round(sum(levels)/len(levels),3),"fairness_variance":e.verification.get("projected_metrics",{}).get("fairness_variance",0) if e.verification else 0,"verifier_intervention_rate":negotiation["rejections"],"renegotiation_count":negotiation["rejections"],"negotiation_rounds":negotiation["rounds"],"invalid_agreement_rate":int(invalid)},"state":e.snapshot(),"negotiation":negotiation}

def run_experiment(req): return {m:run_mode(req,m) for m in ["classical","unverified","verified"]}
