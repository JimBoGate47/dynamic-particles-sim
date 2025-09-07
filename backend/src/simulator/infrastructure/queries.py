from dataclasses import dataclass
from typing import Optional

import torch

from backend.src.common.domain.entities.properties import PhysicalProps, SimulationProps
from backend.src.simulator.domain.queries import InteractionQuery, InteractionResponse


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
