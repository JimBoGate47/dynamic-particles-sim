from dataclasses import dataclass
from typing import List

from src.common.domain.types.particle import Particle2D
from src.common.domain.types.properties import SimulationProperties


@dataclass
class Snaptshot2D:
    particles: List[Particle2D]
    constants: SimulationProperties
    step: int
