from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import Agreement, Proposal, Shipment
from .agents import agent_proposal
from .verifier import verify_agreement
from .policies import classical_policy
from .config import LLM_MODEL

logger = logging.getLogger(__name__)

def build_agreement(proposals):
    chosen = {}; production = []
    for p in proposals:
        for s in p.shipments:
            key = (s.from_node, s.to); chosen[key] = max(chosen.get(key, 0), s.quantity)
        production.extend(p.production)
    return Agreement(
        shipments=[Shipment(**{"from": a, "to": b, "quantity": q}) for (a, b), q in chosen.items()],
        production=production
    )

def negotiate(engine, verified=True):
    feedback = None; log = []; attempts = 0; used_mock = False
    agents = ["supplier", "manufacturer", "distributor", "retailer"]

    max_rounds = engine.experiment.max_negotiation_rounds
    for round_no in range(1, max_rounds + 1):
        proposals = [None] * len(agents)

        # ── Run all 4 agent calls in parallel (4× faster than sequential) ──
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(agent_proposal, name, engine, round_no, feedback): idx
                for idx, name in enumerate(agents)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                name = agents[idx]
                try:
                    p, mocked = fut.result()
                except Exception as exc:
                    logger.error("Agent '%s' raised unexpectedly: %s", name, exc)
                    from .agents import mock_proposal
                    p, mocked = mock_proposal(name, engine, round_no, feedback), True
                used_mock = used_mock or mocked
                proposals[idx] = p
                log.append({
                    "speaker": f"{name.title()} Agent" + (" (Mock)" if mocked else ""),
                    "message": p.reason,
                    "proposal": p.model_dump(by_alias=True)
                })

        agreement = build_agreement(proposals)
        result = verify_agreement(engine, agreement)
        attempts += 1
        log.append({
            "speaker": "Verifier",
            "message": "AGREEMENT VERIFIED" if result.valid else " · ".join(v["constraint"] for v in result.violations),
            "valid": result.valid,
            "violations": result.violations
        })

        if not verified or result.valid:
            engine.mock_agents = used_mock
            return agreement, result, log, {
                "rounds": round_no, "attempts": attempts,
                "max_rounds": max_rounds,
                "rejections": attempts - 1, "status": "executed",
                "agent_model": "mock" if used_mock else LLM_MODEL
            }
        feedback = result.violations

    # Fallback: classical policy after max rounds exhausted
    agreement = classical_policy(engine)
    result = verify_agreement(engine, agreement)
    log.append({"speaker": "System", "message": "Negotiation exhausted; classical emergency policy applied."})
    engine.mock_agents = used_mock
    return agreement, result, log, {
        "rounds": max_rounds, "attempts": attempts, "max_rounds": max_rounds,
        "rejections": attempts, "status": "fallback",
        "agent_model": "mock" if used_mock else LLM_MODEL
    }
