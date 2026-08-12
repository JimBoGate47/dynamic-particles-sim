from dataclasses import dataclass

import torch
from bson import ObjectId

from src.common.domain.entities.properties import PhysicalProps, SimulationProps
from src.simulator.infrastructure.queries import GenericInteractionResponse


@dataclass
class DomainEvent:
    pass


@dataclass
class SimulationStepCompleted(DomainEvent):
    step: int
    step_ordinal: int
    total_steps: int
    batch_id: str | None
    constants_id: ObjectId
    positions: torch.Tensor
    velocity: torch.Tensor
    phys_props: PhysicalProps
    sim_props: SimulationProps
    response: GenericInteractionResponse


@dataclass
class SimulationSnapshotPersisted(DomainEvent):
    snapshot_id: str
    batch_id: str | None
    step: int
