from dataclasses import dataclass

from backend.src.common.domain.interfaces import UseCase
from backend.src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotsRemover(UseCase):
    constants_id: str
    orm_snapshot: SnapshotRepository

    async def execute(self, *args, **kwargs):
        await self.orm_snapshot.delete_with_constants_id(_id=self.constants_id)
