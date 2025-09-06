from dataclasses import dataclass

from bson import ObjectId

from backend.src.common.domain.entities import Snapshot, Particle
from backend.src.common.domain.interfaces import UseCase
from backend.src.common.domain.repositories.constants import ConstantsRepository
from backend.src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotBuilder(UseCase):
    step: int
    constants_id: ObjectId
    particles: list[Particle]
    orm_snapshot: SnapshotRepository
    orm_constants: ConstantsRepository | None = None

    async def execute(self, *args, **kwargs) -> Snapshot | None:
        constants = await self.orm_constants.find_by_id(self.constants_id)
        if not constants:
            raise ValueError(f"No se encontraro constants: {str(self.constants_id)}")

        return await self.orm_snapshot.persist(
            Snapshot(
                step=self.step,
                constants=constants,
                particles=self.particles
            )
        )
