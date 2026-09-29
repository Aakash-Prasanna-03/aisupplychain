from __future__ import annotations
import json, re, time, logging, urllib.request, urllib.error
from .models import Proposal, Shipment
from .config import LLM_ENABLED, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# ── Retry config ─────────────────────────────────────────────────────────────
_MAX_RETRIES = 2
_RETRY_BACKOFF = 1.5

# Detect if we're using the native Gemini REST API (not OpenAI-compat layer)
_GEMINI_NATIVE = "generativelanguage.googleapis.com" in LLM_BASE_URL and "/openai" not in LLM_BASE_URL

def mock_proposal(name, engine, round_no, feedback=None):
    edge = {"supplier": ("supplier", "manufacturer"), "manufacturer": ("manufacturer", "distributor"),
            "distributor": ("distributor", "retailer"), "retailer": ("distributor", "retailer")}[name]
    source, dest = edge; available = engine.nodes[source].inventory
    intensity = {"conservative": .7, "balanced": 1.0, "aggressive": 1.3}[engine.experiment.recovery_aggressiveness]
    if name == "supplier": q = min((70 if round_no == 1 and not feedback else 48) * intensity, max(0, available))
    else: q = min(25 * intensity, max(0, available))
    objectives = {
        "supplier": "Protect upstream supply and recovery capacity.",
        "manufacturer": "Stabilize material flow into production.",
        "distributor": "Protect downstream inventory availability.",
        "retailer": "Preserve customer service continuity.",
    }
    reason = (objectives[name] if not feedback
              else f"Revised shipment to address verifier constraints while continuing to {objectives[name].lower()}")
    return Proposal(proposer=name, shipments=[Shipment(**{"from": source, "to": dest, "quantity": q})], reason=reason)

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
            if exc.code in (401, 403):
                raise RuntimeError(f"LLM auth error {exc.code}: check your API key in .env") from exc
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


def agent_proposal(name, engine, round_no, feedback=None):
    """Use agent's local node context; any API failure falls back to mock."""
    if not LLM_ENABLED:
        return mock_proposal(name, engine, round_no, feedback), True

    node = engine.nodes[name]
    prompt = (
        "Return ONLY a JSON object with keys: proposer, shipments, production, reason. "
        f"You are the {name} supply-chain agent. "
        f"Local state: inventory={node.inventory}, capacity={node.capacity}, "
        f"production_capacity={node.production_capacity}. "
        f"Scenario: {engine.disruption.description}. Effects={[(effect.type, effect.target, effect.magnitude) for effect in engine.disruption.effects]}. "
        f"Affected nodes={engine.disruption.affected_nodes}. Routes={[(route.from_node, route.to_node) for route in engine.disruption.affected_routes]}. "
        f"Duration={engine.disruption.duration} days. Severity={engine.disruption.severity:.0%}. "
        f"Recovery aggressiveness={engine.experiment.recovery_aggressiveness}. "
        f"Round={round_no}. Verifier feedback={feedback or 'none'}. "
        "shipments must be a list of objects each with from, to, quantity (numbers). "
        "production must be an empty list unless you are the manufacturer. "
        "Only propose non-negative quantities on your adjacent supply-chain route."
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
        return mock_proposal(name, engine, round_no, feedback), True
