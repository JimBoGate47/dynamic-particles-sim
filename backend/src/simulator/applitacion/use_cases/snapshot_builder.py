from dataclasses import dataclass

from backend.src.common.domain.entities import Snapshot, Constants, Particle
from backend.src.common.domain.interfaces import UseCase
from backend.src.common.domain.repositories.constants import ConstantsRepository
from backend.src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotBuilder(UseCase):
    step: int
    constants: Constants
    particles: list[Particle]
    orm_snapshot: SnapshotRepository
    orm_constants: ConstantsRepository | None = None

    async def execute(self, *args, **kwargs) -> Snapshot:
        return await self.orm_snapshot.persist(
            Snapshot(
                step=self.step,
                constants=self.constants,
                particles=self.particles
            )
        )
