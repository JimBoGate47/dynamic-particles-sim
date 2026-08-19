from dataclasses import dataclass

from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotsBatchDeleter(UseCase):
    batch_id: str
    snapshot_repository: SnapshotRepository

    async def execute(self, *args, **kwargs) -> bool:
        return await self.snapshot_repository.delete_by_batch_id(self.batch_id)