from dataclasses import dataclass

import torch

from backend.src.common.domain.entities.properties import PhysicalProps
from backend.src.simulator.domain.queries import InteractionQuery


@dataclass
class GenericInteractionQuery(InteractionQuery):
    positions: torch.Tensor
    velocity: torch.Tensor
    beta: float
    phys_props: PhysicalProps
