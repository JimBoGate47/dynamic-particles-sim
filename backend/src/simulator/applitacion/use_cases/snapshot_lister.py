from dataclasses import dataclass

from src.common.domain.entities import Snapshot
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotsLister(UseCase):
    filters: SnapshotsFilter
    snapshot_repository: SnapshotRepository

    async def execute(self, *args, **kwargs) -> list[Snapshot]:
        return await self.snapshot_repository.filter(params=self.filters)
