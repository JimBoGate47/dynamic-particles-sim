from typing import List

from beanie import Document, Link

from backend.src.common.domain.entities import Particle
from backend.src.common.domain.models.constants import ConstantsORM


class SnapshotORM(Document):
    step: int
    constants: Link[ConstantsORM]
    particles: List[Particle]
