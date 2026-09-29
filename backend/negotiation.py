from __future__ import annotations
import logging
from .models import Agreement, Proposal, Shipment
from .agents import agent_proposal
from .verifier import verify_agreement
from .policies import classical_policy
from .config import LLM_MODEL

logger = logging.getLogger(__name__)

def build_agreement(proposals: list[Proposal]) -> Agreement:
    chosen: dict[tuple[str, str], float] = {}
    production = []
    for p in proposals:
        if p is None:
            continue
        for s in p.shipments:
            key = (s.from_node, s.to)
            # Use proposed quantity for each route
            chosen[key] = max(chosen.get(key, 0.0), s.quantity)
        production.extend(p.production)
    return Agreement(
        shipments=[Shipment(**{"from": a, "to": b, "quantity": q}) for (a, b), q in chosen.items()],
        production=production
    )

def negotiate(engine, verified=True):
    feedback = None
    log = []
    attempts = 0
    used_mock = False
    agents = ["supplier", "manufacturer", "distributor", "retailer"]

    max_rounds = engine.experiment.max_negotiation_rounds
    for round_no in range(1, max_rounds + 1):
        proposals: list[Proposal | None] = [None] * len(agents)
        prior_proposals: dict[str, Proposal] = {}

        # Run sequential turn-by-turn counterproposals
        for idx, name in enumerate(agents):
            try:
                p, mocked = agent_proposal(name, engine, round_no, feedback, prior_proposals)
            except Exception as exc:
                logger.error("Agent '%s' proposal error: %s", name, exc)
                from .agents import mock_proposal
                p, mocked = mock_proposal(name, engine, round_no, feedback, prior_proposals), True
            used_mock = used_mock or mocked
            proposals[idx] = p
            prior_proposals[name] = p
            log.append({
                "speaker": f"{name.title()} Agent" + (" (Fallback)" if mocked else ""),
                "message": p.reason,
                "proposal": p.model_dump(by_alias=True),
                "round": round_no,
            })

        agreement = build_agreement([p for p in proposals if p is not None])
        result = verify_agreement(engine, agreement)
        attempts += 1

        if not verified:
            # Unverified mode: executes proposals directly without safety verification gate
            violations_summary = " · ".join(v["constraint"] for v in result.violations) if result.violations else "None"
            log.append({
                "speaker": "Verifier",
                "message": f"UNVERIFIED MODE: Operating constraints bypassed. Agreement executed without safety gate (Safety audit: {'PASSED' if result.valid else 'VIOLATIONS: ' + violations_summary}).",
                "valid": result.valid,
                "violations": result.violations,
                "round": round_no,
            })
            engine.mock_agents = used_mock
            return agreement, result, log, {
                "rounds": round_no,
                "attempts": attempts,
                "max_rounds": max_rounds,
                "rejections": 0,
                "status": "executed",
                "agent_model": "mock" if used_mock else LLM_MODEL,
                "final_valid": result.valid,
            }

        if result.valid:
            # Verified and approved
            log.append({
                "speaker": "Verifier",
                "message": "AGREEMENT VERIFIED: Storage, route, service, and downstream fairness constraints satisfied. Plan is safe to execute.",
                "valid": True,
                "violations": [],
                "round": round_no,
            })
            engine.mock_agents = used_mock
            return agreement, result, log, {
                "rounds": round_no,
                "attempts": attempts,
                "max_rounds": max_rounds,
                "rejections": attempts - 1,
                "status": "executed",
                "agent_model": "mock" if used_mock else LLM_MODEL,
                "final_valid": True,
            }

        # Verifier rejected this round's proposal: record rejection and pass feedback
        violations_summary = " · ".join(v["message"] for v in result.violations)
        log.append({
            "speaker": "Verifier",
            "message": f"REJECTED (Round {round_no}): {violations_summary}. REQUIRED: Revise allocations to satisfy operating constraints.",
            "valid": False,
            "violations": result.violations,
            "round": round_no,
        })
        feedback = result.violations

    # Fallback to classical policy if max rounds exhausted
    agreement = classical_policy(engine)
    result = verify_agreement(engine, agreement)
    log.append({
        "speaker": "System",
        "message": "Negotiation exhausted; classical emergency policy applied and verified.",
        "round": max_rounds,
    })
    engine.mock_agents = used_mock
    return agreement, result, log, {
        "rounds": max_rounds,
        "attempts": attempts,
        "max_rounds": max_rounds,
        "rejections": attempts,
        "status": "fallback",
        "agent_model": "mock" if used_mock else LLM_MODEL,
        "final_valid": result.valid,
    }
