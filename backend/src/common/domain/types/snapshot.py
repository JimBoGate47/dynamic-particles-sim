from dataclasses import dataclass
from typing import List, Optional

from backend.src.common.domain.types.particle import Particle2D
from backend.src.common.domain.types.properties import SimulationProperties


@dataclass
class Snaptshot2D:
    particles: List[Particle2D]
    constants: SimulationProperties
    step: int


@dataclass
class SnapshotParams:
    step: Optional[int] = None
    constants_name: Optional[str] = None
    snapshot_id: Optional[str] = None
