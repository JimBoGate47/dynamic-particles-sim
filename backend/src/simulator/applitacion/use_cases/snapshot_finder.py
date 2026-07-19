from dataclasses import dataclass

from src.common.domain.entities import Snapshot
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository
from src.simulator.applitacion.mixins import SnapshotFinderMixin


@dataclass
class SnapshotFinderById(UseCase, SnapshotFinderMixin):
    snapshot_id: str
    orm_snapshot: SnapshotRepository

    async def execute(self, *args, **kwargs) -> Snapshot:
        snapshot: Snapshot = await self.find_by_id()
        if not snapshot:
            raise ValueError("Not snapshot found")
        return snapshot
