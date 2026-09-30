from __future__ import annotations

from .models import Disruption, ImportedNode


def active(disruption: Disruption, day: int) -> bool:
    return disruption.start_day <= day < disruption.start_day + disruption.duration


def effects(disruption: Disruption, day: int, physical_nodes: list[ImportedNode] | None = None) -> dict:
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

    physical_by_id = {node.id: node for node in (physical_nodes or []) if node.id}
    explicit_physical = {
        node.operating_tier: node.id
        for node in physical_by_id.values()
        if node.id in disruption.affected_nodes
    }

    for effect in disruption.effects:
        kind = effect.type.lower().replace("-", "_").replace(" ", "_")
        target = effect.target
        physical_target = target if target in physical_by_id else explicit_physical.get(target)
        magnitude = max(0.0, min(effect.magnitude * disruption.effect_scale, 1.0))
        if kind in {"capacity_reduction", "production_reduction", "production_shutdown", "supplier_failure", "energy_shortage", "quality_failure"}:
            if target:
                # A physical target such as S1 is applied to its tier's
                # aggregate using capacity weighting. This preserves the
                # current tier-level execution engine without silently
                # broadening a node-specific disruption to every peer.
                if physical_target:
                    target_node = physical_by_id[physical_target]
                    tier = target_node.operating_tier
                    tier_capacity = sum(
                        float(node.capacity) for node in physical_by_id.values()
                        if node.operating_tier == tier
                    )
                    weighted = float(target_node.capacity) / max(tier_capacity, 1.0)
                    result["capacity_factors"][tier] = result["capacity_factors"].get(tier, 1.0) * (1.0 - magnitude * weighted)
                else:
                    result["capacity_factors"][target] = result["capacity_factors"].get(target, 1.0) * (1.0 - magnitude)
            else:
                result["capacity_factor"] *= 1.0 - magnitude
        elif kind in {"throughput_reduction", "handling_delay", "warehouse_failure", "technology_failure", "cyberattack"}:
            if target:
                tier = physical_by_id[physical_target].operating_tier if physical_target else target
                result["throughput_factors"][tier] = result["throughput_factors"].get(tier, 1.0) * (1.0 - magnitude)
        elif kind in {"inventory_loss", "stock_loss", "storage_damage"} and target and day == disruption.start_day:
            tier = physical_by_id[physical_target].operating_tier if physical_target else target
            if physical_target:
                tier_inventory = sum(float(node.inventory) for node in physical_by_id.values() if node.operating_tier == tier)
                inventory_weight = float(physical_by_id[physical_target].inventory) / max(tier_inventory, 1.0)
                magnitude *= inventory_weight
            result["inventory_losses"][tier] = result["inventory_losses"].get(tier, 0.0) + magnitude
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
