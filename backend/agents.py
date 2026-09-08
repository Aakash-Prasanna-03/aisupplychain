from __future__ import annotations
import json, re, urllib.request
from .models import Proposal, Shipment
from .config import LLM_ENABLED, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS, LLM_LOCAL_ENDPOINT

def mock_proposal(name, engine, round_no, feedback=None):
    # Deliberately ambitious opening proposal; feedback forces feasible revision.
    edge={"supplier":("supplier","manufacturer"),"manufacturer":("manufacturer","distributor"),"distributor":("distributor","retailer"),"retailer":("distributor","retailer")}[name]
    source,dest=edge; available=engine.nodes[source].inventory
    if name=="supplier": q=min(70 if round_no==1 and not feedback else 48, max(0,available))
    else: q=min(25,max(0,available))
    objectives={
        "supplier":"Protect upstream supply and recovery capacity.",
        "manufacturer":"Stabilize material flow into production.",
        "distributor":"Protect downstream inventory availability.",
        "retailer":"Preserve customer service continuity.",
    }
    reason=(objectives[name] if not feedback else f"Revised shipment to address verifier constraints while continuing to {objectives[name].lower()}")
    return Proposal(proposer=name, shipments=[Shipment(**{"from":source,"to":dest,"quantity":q})], reason=reason)

def _parse_json(content):
    """Accept strict JSON and occasional fenced JSON from a model."""
    if isinstance(content, dict): return content
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match: raise ValueError("Qwen returned no JSON object")
        return json.loads(match.group(0))

def call_llm(prompt):
    if not LLM_ENABLED or (not LLM_API_KEY and not LLM_LOCAL_ENDPOINT): raise RuntimeError("Qwen credentials unavailable")
    payload={"model":LLM_MODEL,"messages":[{"role":"system","content":"You are a supply-chain negotiation agent. Follow the requested JSON schema exactly."},{"role":"user","content":prompt}],"temperature":0.1}
    # Hosted OpenAI-compatible providers accept response_format. Ollama's
    # compatible endpoint is more reliable when the schema is enforced by the
    # prompt and our parser instead.
    if not LLM_LOCAL_ENDPOINT:
        payload["response_format"]={"type":"json_object"}
    body=json.dumps(payload).encode()
    headers={"Content-Type":"application/json","User-Agent":"signal-chain/1.0"}
    if LLM_API_KEY: headers["Authorization"]=f"Bearer {LLM_API_KEY}"
    req=urllib.request.Request(f"{LLM_BASE_URL.rstrip('/')}/chat/completions",data=body,headers=headers)
    with urllib.request.urlopen(req,timeout=LLM_TIMEOUT_SECONDS) as res: payload=json.loads(res.read())
    try: content=payload["choices"][0]["message"]["content"]
    except (KeyError,IndexError,TypeError) as exc: raise ValueError("Qwen response did not contain a chat completion") from exc
    return _parse_json(content)

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
        if not isinstance(raw, dict): raise ValueError("Qwen proposal must be a JSON object")
        raw["proposer"] = name
        return Proposal.model_validate(raw), False
    except Exception:
        return mock_proposal(name, engine, round_no, feedback), True
