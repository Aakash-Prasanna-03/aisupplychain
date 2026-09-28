from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Literal

NodeId = Literal["supplier", "manufacturer", "distributor", "retailer"]

class Node(BaseModel):
    id: NodeId; name: str; inventory: float; capacity: float; production_capacity: float = 0
    holding_cost: float = 0.2; shortage_cost: float = 4; service_level_target: float = .9
    service_level: float = 1; status: str = "NORMAL"

class Shipment(BaseModel):
    from_node: NodeId = Field(alias="from")
    to: NodeId
    quantity: float = Field(ge=0)
    class Config: populate_by_name = True

class Production(BaseModel):
    node: NodeId; quantity: float = Field(ge=0)

class Proposal(BaseModel):
    proposer: NodeId; shipments: list[Shipment] = []; production: list[Production] = []; reason: str = ""

class Agreement(BaseModel):
    shipments: list[Shipment] = []; production: list[Production] = []; priority_allocations: dict[str, float] = {}

class Disruption(BaseModel):
    type: Literal["supplier_capacity_drop", "route_closure", "demand_spike"] = "supplier_capacity_drop"
    start_day: int = 5; duration: int = 5; severity: float = Field(.4, ge=0, le=1)

NETWORK_ROLE_ALIASES = {
    "supplier": "supplier", "vendor": "supplier", "manufacturer": "manufacturer",
    "factory": "manufacturer", "plant": "manufacturer", "distributor": "distributor",
    "warehouse": "distributor", "distribution_center": "distributor", "retailer": "retailer",
    "store": "retailer", "customer": "retailer",
}

class ImportedNode(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    node_type: str = Field(min_length=1, max_length=40)
    inventory: float = Field(ge=0)
    capacity: float = Field(gt=0)
    production_capacity: float = Field(0, ge=0)
    holding_cost: float = Field(.2, ge=0)
    shortage_cost: float = Field(4, ge=0)
    service_level_target: float = Field(.9, ge=.1, le=1)

    @property
    def operating_tier(self) -> str:
        return NETWORK_ROLE_ALIASES.get(self.node_type.strip().lower(), "")

class NetworkImport(BaseModel):
    nodes: list[ImportedNode] = Field(min_length=4, max_length=500)

    @model_validator(mode="after")
    def validate_operating_tiers(self):
        unknown = sorted({node.node_type for node in self.nodes if not node.operating_tier})
        if unknown:
            allowed = ", ".join(sorted(NETWORK_ROLE_ALIASES))
            raise ValueError(f"Unsupported node_type values: {', '.join(unknown)}. Use one of: {allowed}")
        present = {node.operating_tier for node in self.nodes}
        missing = [tier for tier in ("supplier", "manufacturer", "distributor", "retailer") if tier not in present]
        if missing:
            raise ValueError(f"Imported network needs at least one node for each operating tier. Missing: {', '.join(missing)}")
        return self

class SimulationRequest(BaseModel):
    seed: int = 42; mode: Literal["classical", "unverified", "verified"] = "verified"
    disruption: Disruption = Disruption(); simulation_days: int = Field(12, ge=1, le=60)
    network: NetworkImport | None = None

class VerificationResult(BaseModel):
    valid: bool; violations: list[dict] = []; projected_metrics: dict = {}

class ARDNRuntimeTuning(BaseModel):
    """Bounded controls applied to the active in-memory ARDN implementation."""
    forecast_horizon: int = Field(15, ge=4, le=15)
    service_threshold: float = Field(.9, ge=.7, le=.98)
    cost_interval_multiplier: float = Field(1., ge=.5, le=3.)
    recovery_weight: float = Field(1., ge=0., le=3.)
    cost_weight: float = Field(.35, ge=0., le=3.)
    risk_weight: float = Field(.75, ge=0., le=3.)
    service_loss_weight: float = Field(.75, ge=0., le=3.)
    ood_weight: float = Field(.35, ge=0., le=3.)
    risk_review_threshold: float = Field(.6, ge=.05, le=.95)
    ood_review_threshold: float = Field(15., ge=.1, le=100.)
    enabled_actions: list[Literal["reallocate", "reroute", "delay", "prioritize", "share_capacity", "emergency_source"]] = Field(default_factory=lambda:["reallocate", "reroute", "delay", "prioritize", "share_capacity", "emergency_source"], min_length=1)
