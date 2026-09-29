from __future__ import annotations

from .models import Disruption


def active(disruption: Disruption, day: int) -> bool:
    return disruption.start_day <= day < disruption.start_day + disruption.duration


def effects(disruption: Disruption, day: int) -> dict:
    """Translate extensible LLM effects into the simulator's operational signals."""
    result = {
        "capacity_factor": 1.0,
        "capacity_factors": {},
        "throughput_factors": {},
        "route_closed": False,
        "closed_routes": set(),
        "demand_factor": 1.0,
        "inventory_losses": {},
    }
    if not active(disruption, day):
        return result

    for effect in disruption.effects:
        kind = effect.type.lower().replace("-", "_").replace(" ", "_")
        target = effect.target
        magnitude = max(0.0, min(effect.magnitude * disruption.effect_scale, 1.0))
        if kind in {"capacity_reduction", "production_reduction", "production_shutdown", "supplier_failure", "energy_shortage", "quality_failure"}:
            if target:
                result["capacity_factors"][target] = result["capacity_factors"].get(target, 1.0) * (1.0 - magnitude)
            else:
                result["capacity_factor"] *= 1.0 - magnitude
        elif kind in {"throughput_reduction", "handling_delay", "warehouse_failure", "technology_failure", "cyberattack"}:
            if target:
                result["throughput_factors"][target] = result["throughput_factors"].get(target, 1.0) * (1.0 - magnitude)
        elif kind in {"inventory_loss", "stock_loss", "storage_damage"} and target and day == disruption.start_day:
            result["inventory_losses"][target] = result["inventory_losses"].get(target, 0.0) + magnitude
        elif kind in {"route_closure", "shipping_delay", "transport_disruption"}:
            route = (effect.source, effect.destination)
            if effect.source and effect.destination:
                result["closed_routes"].add(route)
            else:
                result["route_closed"] = True
        elif kind in {"demand_spike", "demand_increase", "demand_change"}:
            result["demand_factor"] *= 1.0 + magnitude
        elif kind in {"demand_drop", "demand_reduction"}:
            result["demand_factor"] *= max(0.0, 1.0 - magnitude)

    return result
