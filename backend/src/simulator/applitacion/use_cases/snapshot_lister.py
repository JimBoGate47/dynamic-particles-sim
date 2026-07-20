from dataclasses import dataclass

from src.common.domain.entities import SnapshotsCollection
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository
from src.common.helpers.snapshot_collection import group_by_batch_id


@dataclass
class SnapshotsLister(UseCase):
    filters: SnapshotsFilter
    snapshot_repository: SnapshotRepository

    async def execute(self, *args, **kwargs) -> list[SnapshotsCollection]:
        snapshots = await self.snapshot_repository.filter(params=self.filters)
        if not snapshots:
            return []
        return group_by_batch_id(snapshots)
