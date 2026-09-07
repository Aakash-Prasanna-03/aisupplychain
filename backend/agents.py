from __future__ import annotations
import json, urllib.request
from .models import Proposal, Shipment
from .config import LLM_ENABLED, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

def mock_proposal(name, engine, round_no, feedback=None):
    # Deliberately ambitious opening proposal; feedback forces feasible revision.
    edge={"supplier":("supplier","manufacturer"),"manufacturer":("manufacturer","distributor"),"distributor":("distributor","retailer"),"retailer":("distributor","retailer")}[name]
    source,dest=edge; available=engine.nodes[source].inventory
    if name=="supplier": q=min(70 if round_no==1 and not feedback else 48, max(0,available))
    else: q=min(25,max(0,available))
    return Proposal(proposer=name, shipments=[Shipment(**{"from":source,"to":dest,"quantity":q})], reason=("Protecting local service and recovery capacity." if not feedback else "Revised to address verifier constraints."))

def call_llm(prompt):
    if not (LLM_ENABLED and LLM_API_KEY): raise RuntimeError("LLM unavailable")
    body=json.dumps({"model":LLM_MODEL,"messages":[{"role":"user","content":prompt}],"response_format":{"type":"json_object"},"temperature":0}).encode()
    req=urllib.request.Request(f"{LLM_BASE_URL.rstrip('/')}/chat/completions",data=body,headers={"Authorization":f"Bearer {LLM_API_KEY}","Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=20) as res: return json.loads(json.loads(res.read())["choices"][0]["message"]["content"])

def agent_proposal(name, engine, round_no, feedback=None):
    """Use only the agent's local node context; any API failure stays demonstrable in mock mode."""
    if not LLM_ENABLED:
        return mock_proposal(name, engine, round_no, feedback), True
    node = engine.nodes[name]
    prompt = ("Return JSON only matching {proposer,shipments,production,reason}. You are the "
              f"{name} supply-chain agent. Local state: inventory={node.inventory}, "
              f"capacity={node.capacity}, production_capacity={node.production_capacity}. "
              f"Round={round_no}. Relevant verifier feedback={feedback or 'none'}. "
              "Propose only a non-negative shipment on your adjacent supply-chain route.")
    try:
        raw = call_llm(prompt)
        raw["proposer"] = name
        return Proposal.model_validate(raw), False
    except Exception:
        return mock_proposal(name, engine, round_no, feedback), True
