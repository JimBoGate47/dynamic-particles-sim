from dataclasses import dataclass
from typing import List, Optional

from backend.src.common.domain.entities import Snapshot, Constants, Particle
from backend.src.common.domain.interfaces import UseCase
from backend.src.common.domain.repositories.constants import ConstantsRepository
from backend.src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotBuilder(UseCase):
    step: int
    constants: Optional[Constants]
    particles: List[Particle]
    orm_snapshot: SnapshotRepository
    orm_constants: Optional[ConstantsRepository] = None
    constants_id: Optional[str] = None

    async def execute(self, *args, **kwargs) -> Snapshot:
        if self.constants_id:
            constants = await self.orm_constants.find_by_id(_id=self.constants_id)
            return await self.get_snapshot(constants)

        return await self.get_snapshot(self.constants)

    async def get_snapshot(self, constants: Constants) -> Snapshot:
        return await self.orm_snapshot.persist(
            Snapshot(
                step=self.step,
                constants=constants,
                particles=self.particles
            )
        )
