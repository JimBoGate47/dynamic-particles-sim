from dataclasses import dataclass

from src.common.domain.entities import Snapshot
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotFinderMixin(object):
    snapshot_id: str
    orm_snapshot: SnapshotRepository

    async def find_by_id(self) -> Snapshot | None:
        return await self.orm_snapshot.find_by_id(_id=self.snapshot_id)
