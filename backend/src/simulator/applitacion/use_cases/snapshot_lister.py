from dataclasses import dataclass

from backend.src.common.domain.entities import Snapshot
from backend.src.common.domain.filters.snapshot import SnapshotsFilter
from backend.src.common.domain.interfaces import UseCase
from backend.src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotsLister(UseCase):
    filters: SnapshotsFilter
    snapshot_repository: SnapshotRepository

    async def execute(self, *args, **kwargs) -> list[Snapshot]:
        return await self.snapshot_repository.filter(params=self.filters)
