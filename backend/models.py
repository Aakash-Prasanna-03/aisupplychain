from __future__ import annotations
from pydantic import BaseModel, Field
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

class SimulationRequest(BaseModel):
    seed: int = 42; mode: Literal["classical", "unverified", "verified"] = "verified"
    disruption: Disruption = Disruption(); simulation_days: int = Field(12, ge=1, le=60)

class VerificationResult(BaseModel):
    valid: bool; violations: list[dict] = []; projected_metrics: dict = {}
