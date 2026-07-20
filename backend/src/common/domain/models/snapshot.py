from typing import List

from beanie import Document, Link

from src.common.domain.entities.particle import Particle
from src.common.domain.models.constants import ConstantsORM


class SnapshotORM(Document):
    step: int
    constants: Link[ConstantsORM]
    particles: List[Particle]
    batch_id: str
