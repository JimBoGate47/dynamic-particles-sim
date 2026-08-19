import asyncio
import io
import json
import zipfile

from src.common.domain.entities.constants import Constants
from src.common.domain.entities.particle import Particle
from src.common.domain.entities.properties import SimulationProps
from src.common.domain.entities.snapshot import Snapshot
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.simulator.applitacion.use_cases.snapshot_batch_zipper import (
    SnapshotsBatchZipper,
)


class FakeSnapshotRepository:
    def __init__(self, snapshots: list[Snapshot]):
        self._snapshots = snapshots
        self.filtered_with: SnapshotsFilter | None = None

    async def filter(self, params: SnapshotsFilter) -> list[Snapshot]:
        self.filtered_with = params
        return [
            s for s in self._snapshots
            if s.batch_id == params.batch_id
        ]


def _run(coro):
    return asyncio.run(coro)


def _sim_props() -> SimulationProps:
    return SimulationProps(
        g=9.0,
        k=10.0,
        min_vel=0.0,
        r_confinement=6.0,
        k_confinement=0.5,
        beta=0.6,
        dt=0.1,
    )


def _snapshot(step: int, batch_id: str) -> Snapshot:
    return Snapshot(
        id=str(step),
        step=step,
        constants=Constants(
            name="sim",
            sim_props=_sim_props(),
        ),
        particles=[Particle(r=[1.0, 2.0], v=[0.0, 0.0], a=[0.0, 0.0])],
        batch_id=batch_id,
    )


class TestSnapshotsBatchZipper:
    def test_builds_zip_with_one_json_per_snapshot(self):
        batch = "batch-1"
        repo = FakeSnapshotRepository([
            _snapshot(step=10, batch_id=batch),
            _snapshot(step=20, batch_id=batch),
        ])

        result = _run(SnapshotsBatchZipper(
            batch_id=batch,
            snapshot_repository=repo,
        ).execute())

        assert result.filename == "snapshot_batch-1.zip"
        with zipfile.ZipFile(io.BytesIO(result.content)) as zf:
            names = sorted(zf.namelist())
            assert names == ["snapshot_10.json", "snapshot_20.json"]
            data = json.loads(zf.read("snapshot_10.json"))
            assert data["step"] == 10
            assert data["batch_id"] == batch
            assert data["particles"][0]["r"] == [1.0, 2.0]

    def test_filters_by_batch_id(self):
        repo = FakeSnapshotRepository([
            _snapshot(step=1, batch_id="batch-a"),
            _snapshot(step=2, batch_id="batch-b"),
        ])

        result = _run(SnapshotsBatchZipper(
            batch_id="batch-b",
            snapshot_repository=repo,
        ).execute())

        assert repo.filtered_with.batch_id == "batch-b"
        with zipfile.ZipFile(io.BytesIO(result.content)) as zf:
            assert zf.namelist() == ["snapshot_2.json"]

    def test_returns_none_when_no_snapshots(self):
        repo = FakeSnapshotRepository([
            _snapshot(step=1, batch_id="batch-a"),
        ])

        result = _run(SnapshotsBatchZipper(
            batch_id="batch-missing",
            snapshot_repository=repo,
        ).execute())

        assert result is None