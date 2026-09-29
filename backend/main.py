import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from .models import SimulationRequest, Disruption, ARDNRuntimeTuning, ScenarioPromptRequest
from .simulator import SimulationEngine
from .policies import classical_policy
from .negotiation import negotiate
from .experiments import run_experiment
from .database import init_db,save_run
from .ardn_service import forecast_for_engine, ardn_configuration, update_ardn_runtime_tuning
from .config import LLM_CONFIGURED, LLM_MODEL
from .scenario_parser import interpret_scenario, apply_experiment_config, resolved_experiment_config, normalize_disruption

app=FastAPI(title="Trust-Verified Agentic Negotiation"); app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://localhost:5174"],allow_methods=["*"],allow_headers=["*"])
runs={}
@app.on_event("startup")
def start(): init_db()
@app.get("/api/health")
def health(): return {"status":"ok","mock_agents":not LLM_CONFIGURED,"llm_enabled":LLM_CONFIGURED,"agent_model":"mock" if not LLM_CONFIGURED else LLM_MODEL}
@app.get("/api/ardn/config")
def ardn_config(): return ardn_configuration()
@app.post("/api/scenario/interpret")
def interpret(req: ScenarioPromptRequest):
    disruption, message, source, fallback_reason = interpret_scenario(req.prompt)
    resolved = resolved_experiment_config(disruption, req.experiment)
    disruption = apply_experiment_config(disruption, req.experiment)
    return {"scenario": disruption.model_dump(), "disruption": disruption.model_dump(), "experiment": resolved.model_dump(), "message": message, "source": source, "fallback_reason": fallback_reason}
@app.put("/api/ardn/config")
def update_ardn_config(tuning:ARDNRuntimeTuning): return update_ardn_runtime_tuning(tuning)
@app.post("/api/simulation/run")
def run_full_simulation(req: SimulationRequest):
    run_id = str(uuid4())
    disruption = apply_experiment_config(normalize_disruption(req.disruption), req.experiment)
    req.disruption = disruption

    # 1. Run full 3-mode comparative experiment on identical seed & disruption
    exp_results = run_experiment(req)
    data = {"id": run_id, "results": exp_results}
    save_run(run_id, data)

    # 2. Extract verified run as the primary live workspace state
    verified_data = exp_results["verified"]
    verified_state = verified_data["state"]

    # 3. Compute ARDN counterfactual forecast evaluated at disruption onset
    engine_for_ardn = SimulationEngine(req.seed, disruption, req.network, req.experiment)
    for _ in range(disruption.start_day - 1):
        engine_for_ardn.step(classical_policy(engine_for_ardn))
    forecast = forecast_for_engine(engine_for_ardn)

    verified_state["forecast"] = forecast
    verified_state["meta"] = verified_data["negotiation"]
    verified_state["id"] = run_id

    runs[run_id] = {
        "engine": None,
        "request": req,
        "experiment": data,
        "state": verified_state,
        "forecast": forecast,
    }

    return {
        "id": run_id,
        "state": verified_state,
        "experiment": data,
        "forecast": forecast,
    }

@app.post("/api/simulation")
def create(req:SimulationRequest):
    disruption=apply_experiment_config(normalize_disruption(req.disruption),req.experiment); e=SimulationEngine(req.seed,disruption,req.network,req.experiment); e.mode=req.mode; id=str(uuid4()); runs[id]={"engine":e,"request":req}; return {"id":id,**e.snapshot()}
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
