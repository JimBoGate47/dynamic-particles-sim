from dataclasses import dataclass
from typing import List

from src.common.domain.entities.particle import Particle
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotParticlesUpdater(UseCase):
    snapshot_id: str
    particles: List[Particle]
    orm_snapshot: SnapshotRepository

    async def execute(self, *args, **kwargs) -> bool:
        return await self.orm_snapshot.update_particles(
            self.snapshot_id,
            self.particles
        )
