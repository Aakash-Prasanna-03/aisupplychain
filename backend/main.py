from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from .models import SimulationRequest, Disruption
from .simulator import SimulationEngine
from .policies import classical_policy
from .negotiation import negotiate
from .experiments import run_experiment
from .database import init_db,save_run
from .ardn_service import forecast_for_engine
from .config import LLM_CONFIGURED, LLM_MODEL

app=FastAPI(title="Trust-Verified Agentic Negotiation"); app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
runs={}
@app.on_event("startup")
def start(): init_db()
@app.get("/api/health")
def health(): return {"status":"ok","mock_agents":not LLM_CONFIGURED,"llm_enabled":LLM_CONFIGURED,"agent_model":"mock" if not LLM_CONFIGURED else LLM_MODEL}
@app.post("/api/simulation")
def create(req:SimulationRequest):
    id=str(uuid4()); e=SimulationEngine(req.seed,req.disruption); e.mode=req.mode; runs[id]={"engine":e,"request":req}; return {"id":id,**e.snapshot()}
@app.get("/api/simulation/{id}")
def get(id:str):
    if id not in runs: raise HTTPException(404,"Unknown simulation")
    return {"id":id,**runs[id]["engine"].snapshot()}
@app.post("/api/simulation/{id}/step")
def step(id:str):
    r=runs.get(id)
    if not r: raise HTTPException(404,"Unknown simulation")
    e=r["engine"]; return e.step(classical_policy(e))
@app.post("/api/simulation/{id}/disruption")
def disruption(id:str,d:Disruption): runs[id]["engine"].disruption=d; return runs[id]["engine"].snapshot()
@app.post("/api/simulation/{id}/negotiate")
def negotiate_route(id:str):
    r=runs.get(id)
    if not r: raise HTTPException(404,"Unknown simulation")
    e=r["engine"]; agreement,result,log,meta=negotiate(e,e.mode=="verified"); e.negotiation=log;e.verification=result.model_dump(); e.step(agreement);return {**e.snapshot(),"meta":meta,"forecast":forecast_for_engine(e)}
@app.get("/api/simulation/{id}/negotiation")
def negotiation(id:str): return {"negotiation":runs[id]["engine"].negotiation,"verification":runs[id]["engine"].verification}
@app.post("/api/experiment")
def experiment(req:SimulationRequest):
    id=str(uuid4()); data={"id":id,"results":run_experiment(req)};save_run(id,data);runs[id]={"experiment":data};return data
@app.get("/api/experiment/{id}")
def get_experiment(id:str): return runs[id]["experiment"]
@app.get("/api/experiment/{id}/comparison")
def comparison(id:str): return {k:v["metrics"] for k,v in runs[id]["experiment"]["results"].items()}
