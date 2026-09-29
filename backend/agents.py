from __future__ import annotations
import json, re, time, logging, urllib.request, urllib.error
from .models import Proposal, Shipment, Production
from .config import LLM_ENABLED, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# ── Retry config ─────────────────────────────────────────────────────────────
_MAX_RETRIES = 2
_RETRY_BACKOFF = 1.5

# Detect if we're using the native Gemini REST API (not OpenAI-compat layer)
_GEMINI_NATIVE = "generativelanguage.googleapis.com" in LLM_BASE_URL and "/openai" not in LLM_BASE_URL

def mock_proposal(name, engine, round_no, feedback=None, prior_proposals=None):
    source_dest = {
        "supplier": ("supplier", "manufacturer"),
        "manufacturer": ("manufacturer", "distributor"),
        "distributor": ("distributor", "retailer"),
        "retailer": ("distributor", "retailer"),
    }[name]
    source, dest = source_dest
    available = engine.nodes[source].inventory
    capacity = engine.nodes[dest].capacity
    dest_inv = engine.nodes[dest].inventory
    intensity = {"conservative": 0.85, "balanced": 1.0, "aggressive": 1.2}[engine.experiment.recovery_aggressiveness]

    severity_pct = int(engine.disruption.severity * 100)

    # In Round 1 without feedback: realistic proposals reflecting the shock
    if round_no == 1 and not feedback:
        if name == "supplier":
            # In round 1, propose steady flow; if aggressive, surge
            base_q = 26 if engine.experiment.recovery_aggressiveness == "aggressive" else 18
            q = min(base_q * intensity, available)
            reason = f"Facing {severity_pct}% upstream disruption. Proposing {q:.1f} units to Manufacturer to support downstream flow while utilizing available buffer inventory ({available:.0f} units)."
        elif name == "manufacturer":
            inbound = prior_proposals.get("supplier").shipments[0].quantity if (prior_proposals and "supplier" in prior_proposals and prior_proposals["supplier"].shipments) else 18.0
            q = min(18 * intensity, available + inbound * 0.4)
            reason = f"Acknowledged {inbound:.1f} units inbound from Supplier. Scheduling production and proposing {q:.1f} units outbound to Distributor to balance material flow."
        elif name == "distributor":
            inbound = prior_proposals.get("manufacturer").shipments[0].quantity if (prior_proposals and "manufacturer" in prior_proposals and prior_proposals["manufacturer"].shipments) else 18.0
            q = min(18 * intensity, available)
            reason = f"Allocating {q:.1f} units to Retailer based on inbound {inbound:.1f} units from Manufacturer to mitigate downstream stockout risk."
        else: # retailer
            inbound = prior_proposals.get("distributor").shipments[0].quantity if (prior_proposals and "distributor" in prior_proposals and prior_proposals["distributor"].shipments) else 18.0
            demand_est = engine.last_demand or 20
            q = min(inbound, available)
            reason = f"Customer demand projected at ~{demand_est:.0f} units. Confirmed {inbound:.1f} units inbound from Distributor to maintain full service target."
    else:
        # Revised proposal addressing verifier constraints
        if name == "supplier":
            safe_q = max(8.0, min(18.0 * intensity, available, capacity - dest_inv))
            q = safe_q
            reason = f"Revising proposal per Verifier constraints: Capping shipment to {q:.1f} units to maintain Manufacturer inventory within safe storage capacity ({capacity:.0f} units)."
        elif name == "manufacturer":
            inbound = prior_proposals.get("supplier").shipments[0].quantity if (prior_proposals and "supplier" in prior_proposals and prior_proposals["supplier"].shipments) else 18.0
            q = max(10.0, min(18.0 * intensity, available + inbound * 0.5))
            reason = f"Aligning outbound shipment to {q:.1f} units to Distributor, maintaining balanced downstream allocation and fairness."
        elif name == "distributor":
            inbound = prior_proposals.get("manufacturer").shipments[0].quantity if (prior_proposals and "manufacturer" in prior_proposals and prior_proposals["manufacturer"].shipments) else 18.0
            q = max(10.0, min(18.0 * intensity, available))
            reason = f"Reallocating {q:.1f} units to Retailer to satisfy minimum service level threshold without violating downstream fairness variance."
        else: # retailer
            inbound = prior_proposals.get("distributor").shipments[0].quantity if (prior_proposals and "distributor" in prior_proposals and prior_proposals["distributor"].shipments) else 18.0
            demand_est = engine.last_demand or 20
            q = min(inbound, available)
            reason = f"Confirmed revised allocation of {inbound:.1f} units. Total available stock ({available + inbound:.1f} units) fulfills projected demand with 0% stockout risk."

    shipments = [Shipment(**{"from": source, "to": dest, "quantity": round(q, 1)})]
    production = [Production(node="manufacturer", quantity=round(q, 1))] if name == "manufacturer" else []
    return Proposal(proposer=name, shipments=shipments, production=production, reason=reason)

def _parse_json(content):
    """Accept strict JSON and occasional fenced JSON from a model."""
    if isinstance(content, dict): return content
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match: raise ValueError("LLM returned no JSON object")
        return json.loads(match.group(0))

def _call_gemini_native(prompt: str) -> dict:
    """Call the native Gemini generateContent REST API directly."""
    url = f"{LLM_BASE_URL.rstrip('/')}/models/{LLM_MODEL}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
        "systemInstruction": {
            "parts": [{"text": "You are a supply-chain negotiation agent. Return ONLY valid JSON. No prose, no markdown fences."}]
        }
    }).encode()
    # Gemini native API uses X-goog-api-key header (not Authorization: Bearer)
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "X-goog-api-key": LLM_API_KEY
    })
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as res:
        resp = json.loads(res.read())
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Gemini response structure: {list(resp.keys())}") from exc

def _call_openai_compat(prompt: str) -> str:
    """Call any OpenAI-compatible /chat/completions endpoint."""
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a supply-chain negotiation agent. Return ONLY valid JSON. No prose, no markdown fences."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "signal-chain/1.0"}
    if LLM_API_KEY: headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as res:
        resp = json.loads(res.read())
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response missing chat completion content") from exc

def call_llm(prompt: str) -> dict:
    """Call the configured LLM with retry/backoff on transient errors."""
    if not LLM_ENABLED or (not LLM_API_KEY):
        raise RuntimeError("LLM credentials unavailable – set QWEN_API_KEY and LLM_ENABLED=true in .env")

    last_exc = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            if _GEMINI_NATIVE:
                content = _call_gemini_native(prompt)
            else:
                content = _call_openai_compat(prompt)
            logger.info("LLM call succeeded on attempt %d/%d (native=%s)", attempt + 1, 1 + _MAX_RETRIES, _GEMINI_NATIVE)
            return _parse_json(content)

        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")[:300]
            logger.warning("LLM HTTP %s on attempt %d: %s", exc.code, attempt + 1, body_text)
            last_exc = exc
            if exc.code in (401, 403, 429):
                raise RuntimeError(f"LLM API error {exc.code} (e.g. quota/auth): {body_text}") from exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("LLM network error on attempt %d: %s", attempt + 1, exc)
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))

        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM bad JSON: %s", exc)
            raise

    raise RuntimeError(f"LLM failed after {1 + _MAX_RETRIES} attempts") from last_exc


def agent_proposal(name, engine, round_no, feedback=None, prior_proposals=None):
    """Use agent's local node context; any API failure falls back to mock."""
    if not LLM_ENABLED:
        return mock_proposal(name, engine, round_no, feedback, prior_proposals), True

    node = engine.nodes[name]
    prior_summary = ""
    if prior_proposals:
        prior_summary = "Prior proposals in this round: " + "; ".join(
            f"{p.proposer} proposed {s.quantity} units from {s.from_node} to {s.to} (reason: {p.reason})"
            for p in prior_proposals.values() for s in p.shipments
        )

    prompt = (
        "Return ONLY a JSON object with keys: proposer, shipments, production, reason. "
        f"You are the {name} supply-chain agent in a multi-agent negotiation. "
        f"Local state: inventory={node.inventory:.1f}, capacity={node.capacity:.1f}, "
        f"production_capacity={node.production_capacity:.1f}. "
        f"Scenario: {engine.disruption.description}. Duration={engine.disruption.duration} days. Severity={engine.disruption.severity:.0%}. "
        f"Recovery aggressiveness={engine.experiment.recovery_aggressiveness}. "
        f"Round={round_no}. Verifier feedback={feedback or 'none'}. "
        f"{prior_summary}. "
        "shipments must be a list of objects each with from, to, quantity (numbers). "
        "production must be an empty list unless you are the manufacturer. "
        "Only propose non-negative quantities on your adjacent supply-chain route. "
        "Respond constructively to upstream/downstream proposals and verifier feedback in your reason."
    )
    try:
        raw = call_llm(prompt)
        if not isinstance(raw, dict):
            raise ValueError("LLM proposal must be a JSON object")
        raw["proposer"] = name
        proposal = Proposal.model_validate(raw)
        logger.info("Agent '%s' used real LLM (round %d)", name, round_no)
        return proposal, False
    except Exception as exc:
        logger.error("Agent '%s' LLM failed → mock. Reason: %s", name, exc)
        return mock_proposal(name, engine, round_no, feedback, prior_proposals), True
