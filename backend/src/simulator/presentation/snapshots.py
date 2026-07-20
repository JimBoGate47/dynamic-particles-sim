from config.database import db_connection
from src.common.domain.entities import Snapshot
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.snapshot_lister import SnapshotsLister


async def list_snapshots(constants_name: str) -> list[dict]:
    async with db_connection():
        snapshots: list[Snapshot] = await SnapshotsLister(
            filters=SnapshotsFilter(
                constants_name=constants_name,
            ),
            snapshot_repository=ORMSnapshotRepository(),
        ).execute()
        if not snapshots:
            raise ValueError("No snapshots found")

        return [
            snapshot.model_dump(mode="json")
            for snapshot in snapshots
        ]
