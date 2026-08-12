from dataclasses import dataclass

import uuid7

from bson import ObjectId

from src.common.domain.entities import Snapshot
from src.common.domain.entities.particle import Particle
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.constants import ConstantsRepository
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotBuilder(UseCase):
    step: int
    constants_id: ObjectId
    particles: list[Particle]
    orm_snapshot: SnapshotRepository
    batch_id: str | None = None
    orm_constants: ConstantsRepository | None = None
    metadata: dict | None = None

    async def execute(self, *args, **kwargs) -> Snapshot | None:
        constants = await self.orm_constants.find_by_id(self.constants_id)
        if not constants:
            raise ValueError(f"No se encontraron constants: {str(self.constants_id)}")

        batch_id_val = self.batch_id if self.batch_id else str(uuid7.create())

        return await self.orm_snapshot.persist(
            Snapshot(
                step=self.step,
                constants=constants,
                particles=self.particles,
                batch_id=batch_id_val,
                metadata=self.metadata or {},
            )
        )
