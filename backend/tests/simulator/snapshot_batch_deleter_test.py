import asyncio

from src.common.domain.entities.constants import Constants
from src.common.domain.entities.particle import Particle
from src.common.domain.entities.properties import SimulationProps
from src.common.domain.entities.snapshot import Snapshot
from src.simulator.applitacion.use_cases.snapshot_batch_deleter import (
    SnapshotsBatchDeleter,
)


class FakeSnapshotRepository:
    def __init__(self):
        self.deleted_with: str | None = None
        self._result = True

    async def delete_by_batch_id(self, batch_id: str) -> bool:
        self.deleted_with = batch_id
        return self._result


def _run(coro):
    return asyncio.run(coro)


class TestSnapshotsBatchDeleter:
    def test_deletes_by_batch_id(self):
        repo = FakeSnapshotRepository()

        deleted = _run(SnapshotsBatchDeleter(
            batch_id="batch-1",
            snapshot_repository=repo,
        ).execute())

        assert repo.deleted_with == "batch-1"
        assert deleted is True

    def test_returns_false_when_nothing_deleted(self):
        repo = FakeSnapshotRepository()
        repo._result = False

        deleted = _run(SnapshotsBatchDeleter(
            batch_id="batch-missing",
            snapshot_repository=repo,
        ).execute())

        assert deleted is False