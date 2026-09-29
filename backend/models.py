from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Literal

NodeId = Literal["supplier", "manufacturer", "distributor", "retailer"]

class Node(BaseModel):
    id: NodeId; name: str; inventory: float; capacity: float; production_capacity: float = 0
    holding_cost: float = 0.2; shortage_cost: float = 4; service_level_target: float = .9
    service_level: float = 1; status: str = "NORMAL"
    status_detail: str = "Operating within nominal limits"
    fulfillment_rate: float = 1.0
    buffer_days: float = 0.0

class Shipment(BaseModel):
    from_node: NodeId = Field(alias="from")
    to: NodeId
    quantity: float = Field(ge=0)
    class Config: populate_by_name = True

class Production(BaseModel):
    node: NodeId = "manufacturer"; quantity: float = Field(ge=0)

class Proposal(BaseModel):
    proposer: NodeId; shipments: list[Shipment] = []; production: list[Production] = []; reason: str = ""

class Agreement(BaseModel):
    shipments: list[Shipment] = []; production: list[Production] = []; priority_allocations: dict[str, float] = {}

class Route(BaseModel):
    from_node: NodeId | None = Field(None, alias="from")
    to_node: NodeId | None = Field(None, alias="to")
    source: NodeId | None = None
    destination: NodeId | None = None
    class Config: populate_by_name = True

    @model_validator(mode="after")
    def normalize_names(self):
        if self.from_node is None:
            self.from_node = self.source
        if self.to_node is None:
            self.to_node = self.destination
        if self.from_node is None or self.to_node is None:
            raise ValueError("Route requires source/from and destination/to")
        return self

class DisruptionEffect(BaseModel):
    type: str = Field(min_length=1, max_length=80)
    target: NodeId | None = None
    source: NodeId | None = None
    destination: NodeId | None = None
    magnitude: float = Field(.4, ge=0, le=1)
    description: str = ""

class Disruption(BaseModel):
    description: str = ""
    affected_nodes: list[NodeId] = Field(default_factory=list)
    affected_routes: list[Route] = Field(default_factory=list)
    effects: list[DisruptionEffect] = Field(default_factory=list)
    start_day: int = Field(1, ge=1, le=60)
    duration: int = Field(5, ge=1, le=60)
    severity: float = Field(.4, ge=0, le=1)
    effect_scale: float = Field(1.0, ge=.1, le=10.0)
    normalization_warnings: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)

class ExperimentConfig(BaseModel):
    severity: int | None = Field(None, ge=10, le=100)
    disruption_duration: int | None = Field(None, ge=1, le=7)
    simulation_horizon: int = Field(12, ge=4, le=14)
    max_negotiation_rounds: int = Field(4, ge=1, le=4)
    recovery_aggressiveness: Literal["conservative", "balanced", "aggressive"] = "balanced"

class ScenarioPromptRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1200)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

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
    disruption: Disruption = Field(default_factory=Disruption); simulation_days: int = Field(12, ge=1, le=60)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    network: NetworkImport | None = None

class VerificationResult(BaseModel):
    valid: bool; violations: list[dict] = []; projected_metrics: dict = {}
    decision: str = "APPROVED"
    audit_trail: list[dict] = []

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
