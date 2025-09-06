import asyncio
from pprint import pprint

from backend.src.common.domain.entities import Snapshot
from backend.src.common.domain.filters.snapshot import SnapshotsFilter
from backend.src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from backend.src.simulator.applitacion.use_cases.snapshot_lister import SnapshotsLister
from config.database import db_connection


async def main():
    async with db_connection():
        snapshots: list[Snapshot] = await SnapshotsLister(
            filters=SnapshotsFilter(
                constants_name="nombre2",
            ),
            snapshot_repository=ORMSnapshotRepository(),
        ).execute()
        pprint(snapshots)


asyncio.run(main())
