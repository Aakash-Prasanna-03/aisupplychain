from __future__ import annotations

import re

from .agents import call_llm
from .config import LLM_CONFIGURED
from .models import Disruption, DisruptionEffect, ExperimentConfig, Route

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
SEVERITY_WORDS = {"critical": .9, "catastrophic": .9, "severe": .8, "highly": .7, "significantly": .7, "substantially": .7, "major": .7, "high": .7, "medium": .45, "moderate": .45, "slightly": .25, "minor": .25, "low": .25}
VALID_EDGES = {("supplier", "manufacturer"), ("manufacturer", "distributor"), ("distributor", "retailer")}
EFFECT_ALIASES = {"production_reduction": "capacity_reduction", "production_shutdown": "capacity_reduction", "factory_fire": "capacity_reduction", "warehouse_failure": "throughput_reduction", "cyberattack": "throughput_reduction", "technology_failure": "throughput_reduction", "transport_disruption": "shipping_delay", "road_closure": "route_closure", "port_closure": "route_closure", "stock_loss": "inventory_loss", "storage_damage": "inventory_loss", "demand_spike": "demand_increase"}
SUPPORTED_EFFECTS = {"capacity_reduction", "inventory_loss", "route_closure", "throughput_reduction", "demand_increase", "demand_drop", "shipping_delay"}


def _number(value: str) -> int:
    cleaned = re.sub(r"(st|nd|rd|th)$", "", value.lower().strip())
    return int(cleaned) if cleaned.isdigit() else NUMBER_WORDS.get(cleaned, 0)


def _fallback(prompt: str) -> Disruption:
    text = prompt.lower()
    physical_ids = [value.upper() for value in re.findall(r"\b[A-Z]{1,4}\d+\b", prompt)]
    nodes = []
    for node, words in {
        "supplier": ("supplier", "vendor", "raw material"),
        "manufacturer": ("manufacturer", "factory", "plant", "production"),
        "distributor": ("distributor", "warehouse", "distribution", "logistics"),
        "retailer": ("retailer", "store", "customer", "consumer"),
    }.items():
        if any(word in text for word in words):
            nodes.append(node)
    target = nodes[0] if nodes else "supplier"
    physical_target = physical_ids[0] if physical_ids else target
    explicit_percentages = [float(value) / 100 for value in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", text)]
    magnitude = max(explicit_percentages) if explicit_percentages else next((value for word, value in SEVERITY_WORDS.items() if word in text), .4)
    number = r"\d{1,2}(?:st|nd|rd|th)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    start_match = re.search(rf"(?:day|on)\s*({number})", text) or re.search(rf"\b({number})\s*(?:st|nd|rd|th)\b", text)
    duration_match = re.search(rf"(?:for|last(?:s|ing)?|duration(?: of)?|to)\s*({number})\s*(?:days?|d)", text) or re.search(rf"\b({number})\s*(?:days?|d)\b", text)
    start_day = _number(start_match.group(1)) if start_match else 1
    duration = _number(duration_match.group(1)) if duration_match else 5
    effects = []
    routes = []
    route_source, route_destination = "supplier", "manufacturer"
    if "manufacturer" in text and "distributor" in text:
        route_source, route_destination = "manufacturer", "distributor"
    elif "distributor" in text and ("to the distributor" in text or "to distributor" in text):
        route_source, route_destination = "manufacturer", "distributor"
    elif "distributor" in text and "retailer" in text:
        route_source, route_destination = "distributor", "retailer"
    if any(word in text for word in ("route", "road", "port", "transport", "shipping", "delivery", "border", "strike", "blocked", "closure")):
        route_magnitude = explicit_percentages[1] if len(explicit_percentages) > 1 else 0.0
        effects.append(DisruptionEffect(type="route_closure", source=route_source, destination=route_destination, magnitude=route_magnitude, description="Transport disruption"))
        routes.append(Route(**{"from": route_source, "to": route_destination}))
        nodes.extend([route_source, route_destination])
    if any(word in text for word in ("inventory", "stock", "flood", "damaged", "damage")) and target in {"distributor", "retailer", "supplier", "manufacturer"}:
        effects.append(DisruptionEffect(type="inventory_loss", target=physical_target, magnitude=explicit_percentages[0] if explicit_percentages else magnitude, description="Inventory damage"))
    if any(word in text for word in ("demand", "orders", "customers", "sales", "promotion", "surge", "spike", "increase")):
        effects.append(DisruptionEffect(type="demand_increase", target="retailer", magnitude=explicit_percentages[-1] if explicit_percentages else magnitude, description="Demand increase"))
    if any(word in text for word in ("cyber", "warehouse system", "erp", "handling", "slow", "technology")):
        effects.append(DisruptionEffect(type="throughput_reduction", target=physical_target, magnitude=explicit_percentages[-1] if explicit_percentages else magnitude, description="Operational throughput reduction"))
    if any(word in text for word in ("factory", "production", "machine", "equipment", "manufacturing", "fire", "labor strike", "energy", "contamination")) and not any(effect.type == "inventory_loss" for effect in effects):
        effects.append(DisruptionEffect(type="capacity_reduction", target=physical_target, magnitude=explicit_percentages[0] if explicit_percentages else magnitude, description="Production capacity reduction"))
    if not effects:
        effects.append(DisruptionEffect(type="capacity_reduction", target=physical_target, magnitude=magnitude, description="Operational capacity reduction"))
    return Disruption(description=prompt, affected_nodes=list(dict.fromkeys((nodes or [target]) + physical_ids)), affected_routes=routes, effects=effects, start_day=max(1, min(start_day, 60)), duration=max(1, min(duration, 60)), severity=max(0, min(magnitude, 1)))


def normalize_disruption(disruption: Disruption) -> Disruption:
    """Convert LLM vocabulary into bounded, executable simulator effects."""
    warnings = list(disruption.normalization_warnings)
    affected = list(disruption.affected_nodes)
    route_defaults = [(route.from_node, route.to_node) for route in disruption.affected_routes]
    normalized = []
    for effect in disruption.effects:
        kind = effect.type.lower().replace("-", "_").replace(" ", "_")
        mapped = EFFECT_ALIASES.get(kind, kind)
        if mapped not in SUPPORTED_EFFECTS:
            mapped = "throughput_reduction" if any(word in (kind + " " + effect.description.lower()) for word in ("cyber", "system", "warehouse", "technology", "slow")) else "capacity_reduction"
            warnings.append(f"Normalized unsupported effect '{effect.type}' to '{mapped}'.")
        target = effect.target
        if mapped in {"capacity_reduction", "inventory_loss", "throughput_reduction"} and target is None:
            target = affected[0] if affected else ("distributor" if mapped == "throughput_reduction" else "supplier")
            warnings.append(f"Inferred {mapped} target as {target}.")
        source, destination = effect.source, effect.destination
        if mapped in {"route_closure", "shipping_delay"}:
            if (source, destination) not in VALID_EDGES:
                source, destination = route_defaults[0] if route_defaults and route_defaults[0] in VALID_EDGES else ("manufacturer", "distributor")
                warnings.append(f"Inferred valid route {source} → {destination} for {mapped}.")
        magnitude = max(0.0, min(float(effect.magnitude), 1.0))
        normalized.append(DisruptionEffect(type=mapped, target=target, source=source, destination=destination, magnitude=magnitude, description=effect.description or effect.type))
        if target and target not in affected:
            affected.append(target)
    return Disruption.model_validate({**disruption.model_dump(), "affected_nodes": list(dict.fromkeys(affected)), "effects": normalized, "normalization_warnings": warnings})


def apply_experiment_config(disruption: Disruption, config: ExperimentConfig) -> Disruption:
    data = disruption.model_dump()
    if config.disruption_duration is not None:
        data["duration"] = config.disruption_duration
    if config.severity is not None:
        previous_severity = max(disruption.severity, .1)
        data["severity"] = config.severity / 100
        data["effect_scale"] = min(10.0, max(.1, data["severity"] / previous_severity))
    return Disruption.model_validate(data)

def resolved_experiment_config(disruption: Disruption, config: ExperimentConfig) -> ExperimentConfig:
    """Return the effective values for display without losing explicit overrides."""
    data = config.model_dump()
    if data["severity"] is None:
        data["severity"] = round(disruption.severity * 100)
    if data["disruption_duration"] is None:
        data["disruption_duration"] = min(7, max(1, disruption.duration))
    return ExperimentConfig.model_validate(data)


def interpret_scenario(prompt: str, config: ExperimentConfig | None = None) -> tuple[Disruption, str, str, str | None]:
    disruption = None
    source = "fallback"
    fallback_reason = None
    if LLM_CONFIGURED:
        instruction = (
            "Interpret this arbitrary supply-chain scenario. Return ONLY valid JSON with keys: description, affected_nodes, affected_routes, effects, start_day, duration, severity. "
            "affected_nodes may use supplier, manufacturer, distributor, retailer, or physical IDs such as S1/M1 when present in the scenario. Preserve physical IDs exactly. affected_routes contain from and to. "
            "effects is an array; each effect has type, optional target/source/destination, magnitude from 0 to 1, and description. "
            "Use operational effects such as capacity_reduction, inventory_loss, route_closure, demand_increase, demand_drop, throughput_reduction, storage_reduction, or shipping_delay. "
            "Represent every simultaneous event instead of rejecting unusual wording. Infer start_day=1 and duration=5 when omitted. "
            "User scenario: " + prompt
        )
        try:
            disruption = Disruption.model_validate(call_llm(instruction))
            explicit_percentages = [float(value) / 100 for value in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", prompt)]
            if explicit_percentages:
                data = disruption.model_dump()
                data["severity"] = max(explicit_percentages)
                disruption = Disruption.model_validate(data)
            source = "llm"
        except ValueError:
            fallback_reason = "invalid_json_or_schema"
            disruption = None
        except RuntimeError as exc:
            fallback_reason = "gemini_unavailable" if "failed after" in str(exc).lower() or "network" in str(exc).lower() else "llm_error"
            disruption = None
        except Exception:
            fallback_reason = "llm_error"
            disruption = None
    if disruption is None:
        disruption = _fallback(prompt)
    disruption = normalize_disruption(disruption)
    if config:
        disruption = apply_experiment_config(disruption, config)
    affected = ", ".join(disruption.affected_nodes) or "the supply chain"
    effects = "; ".join(effect.description or effect.type.replace("_", " ") for effect in disruption.effects)
    return disruption, f"Understood: {disruption.description or prompt}. Affected: {affected}. Effect: {effects}. Estimated impact: {round(disruption.severity * 100)}%. Starts day {disruption.start_day} and lasts {disruption.duration} days.", source, fallback_reason
