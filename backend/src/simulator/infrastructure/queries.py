from dataclasses import dataclass
from typing import Optional

import torch

from src.common.domain.entities.properties import PhysicalProps, SimulationProps
from src.simulator.domain.queries import (
    InteractionQuery,
    InteractionResponse,
    RestrictionResponse,
    RestrictionQuery,
)


@dataclass
class GenericInteractionQuery(InteractionQuery):
    positions: torch.Tensor
    velocity: torch.Tensor
    sim_props: SimulationProps
    phys_props: PhysicalProps


@dataclass
class GenericInteractionResponse(InteractionResponse):
    positions: Optional[torch.Tensor] = None
    velocity: Optional[torch.Tensor] = None
    acceleration: Optional[torch.Tensor] = None


@dataclass
class PositionRestrictionQuery(RestrictionQuery):
    old_positions: torch.Tensor
    new_positions: torch.Tensor


@dataclass
class PositionRestrictionResponse(RestrictionResponse):
    new_positions: torch.Tensor
