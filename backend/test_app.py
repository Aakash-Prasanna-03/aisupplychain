from backend.simulator import SimulationEngine
from backend.models import Agreement,Shipment,SimulationRequest
from backend.verifier import verify_agreement
from backend.negotiation import negotiate
from backend.experiments import run_mode

def test_capacity_violation():
 e=SimulationEngine(); r=verify_agreement(e,Agreement(shipments=[Shipment(**{"from":"supplier","to":"manufacturer","quantity":101})])); assert not r.valid
def test_negative_inventory():
 e=SimulationEngine(); r=verify_agreement(e,Agreement(shipments=[Shipment(**{"from":"distributor","to":"retailer","quantity":41})])); assert not r.valid
def test_valid_agreement():
 e=SimulationEngine(); assert verify_agreement(e,Agreement(shipments=[Shipment(**{"from":"supplier","to":"manufacturer","quantity":10})])).valid
def test_retry():
 e=SimulationEngine(); _,_,_,meta=negotiate(e,True); assert meta["rejections"]>=1
def test_modes():
 req=SimulationRequest(); assert run_mode(req,"classical")["state"]["day"]==12; assert run_mode(req,"unverified")["negotiation"]["status"]=="executed"; assert run_mode(req,"verified")["negotiation"]["rejections"]>=1
def test_seed():
 a=SimulationEngine(1);b=SimulationEngine(1);a.step();b.step();assert a.demands==b.demands
