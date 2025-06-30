from dataclasses import dataclass
from typing import List, Optional

from backend.src.common.domain.entities import Snapshot
from backend.src.common.domain.interfaces import UseCase
from backend.src.common.domain.repositories.snapshot import SnapshotRepository
from backend.src.common.domain.types.snapshot import SnapshotParams


@dataclass
class SnapshotsLister(UseCase):
    params: SnapshotParams
    orm_snapshot: SnapshotRepository

    async def execute(self, *args, **kwargs) -> Optional[List[Snapshot]]:
        return await self.orm_snapshot.filter_by_params(params=self.params)


@dataclass
class SnapshotsFinder(UseCase):
    snapshot_id: str
    orm_snapshot: SnapshotRepository

    async def execute(self, *args, **kwargs) -> Optional[Snapshot]:
        return await self.orm_snapshot.find_by_id(_id=self.snapshot_id)
