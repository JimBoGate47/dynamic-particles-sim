from bson import ObjectId

from config.database import db_connection
from src.common.domain.entities.particle import Particle
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from src.simulator.applitacion.use_cases.snapshot_lister import SnapshotsLister


async def list_snapshots(constants_name: str) -> list[dict]:
    async with db_connection():
        collections = await SnapshotsLister(
            filters=SnapshotsFilter(
                constants_name=constants_name,
            ),
            snapshot_repository=ORMSnapshotRepository(),
        ).execute()
        return [col.model_dump(mode="json") for col in collections]


async def create_snapshot(data: dict) -> dict:
    async with db_connection():
        snapshot = await SnapshotBuilder(
            step=data["step"],
            constants_id=ObjectId(data["constants_id"]),
            particles=[Particle(**p) for p in data.get("particles", [])],
            batch_id=data.get("batch_id", None),
            orm_snapshot=ORMSnapshotRepository(),
            orm_constants=ORMConstantsRepository(),
        ).execute()
        return snapshot.model_dump(mode="json")
