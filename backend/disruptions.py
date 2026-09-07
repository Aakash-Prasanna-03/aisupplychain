from .models import Disruption

def active(disruption: Disruption, day: int) -> bool:
    return disruption.start_day <= day < disruption.start_day + disruption.duration

def effects(disruption: Disruption, day: int) -> dict:
    if not active(disruption, day): return {"capacity_factor": 1, "route_closed": False, "demand_factor": 1}
    return {
        "capacity_factor": 1-disruption.severity if disruption.type == "supplier_capacity_drop" else 1,
        "route_closed": disruption.type == "route_closure",
        "demand_factor": 1+disruption.severity if disruption.type == "demand_spike" else 1,
    }
