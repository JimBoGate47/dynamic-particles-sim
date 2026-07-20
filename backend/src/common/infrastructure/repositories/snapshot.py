from typing import List, Optional

from beanie import PydanticObjectId

from src.common.domain.entities import Snapshot, SnapshotsCollection
from src.common.domain.entities.particle import Particle
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.domain.models.constants import ConstantsORM
from src.common.domain.models.snapshot import SnapshotORM
from src.common.domain.repositories.snapshot import SnapshotRepository
from src.common.infrastructure.builders.snapshot import build_snapshot





class ORMSnapshotRepository(SnapshotRepository):
    async def filter(self, params: SnapshotsFilter) -> list[Snapshot]:
        filters = {}
        if params.snapshot_id:
            filters["_id"] = params.snapshot_id
        if params.step:
            filters["step"] = params.step
        if params.constants_name:
            filters["constants.name"] = params.constants_name
        if params.batch_id:
            filters["batch_id"] = params.batch_id

        if not filters:
            return None

        snapshots = await SnapshotORM.find(filters, fetch_links=True).to_list()

        return [build_snapshot(snapshot) for snapshot in snapshots]

    async def update_particles(self, _id, particles: List[Particle]) -> bool:
        snapshot_orm = await SnapshotORM.get(_id)
        if snapshot_orm is None:
            return False
        snapshot_orm.particles = particles
        await snapshot_orm.save()
        return True

    async def find_by_id(self, _id, fetch_links=False) -> Optional[Snapshot]:
        snapshot_orm = await SnapshotORM.get(_id, fetch_links=fetch_links)
        if snapshot_orm is None:
            return None
        return build_snapshot(snapshot_orm)

    async def find_with_constants_id(self, _id: str, fetch_links=False) -> Optional[List[Snapshot]]:
        snapshots_orm = await SnapshotORM.find(
            SnapshotORM.constants.id == PydanticObjectId(_id),
        ).to_list()
        if snapshots_orm is None:
            return None

        return [
            build_snapshot(snapshot_orm)
            for snapshot_orm in snapshots_orm
        ]

    async def delete_with_constants_id(self, _id):
        await SnapshotORM.find(
            SnapshotORM.constants.id == PydanticObjectId(_id),
        ).delete()

    async def find_particles(self):
        snapshots = await SnapshotORM.find(fetch_links=True).to_list()
        return [
            build_snapshot(snapshot)
            for snapshot in snapshots
        ]

    async def persist_with_constants_id(
            self, constants_id: str,
            snapshot: Snapshot,
            fetch_links: bool = False
    ) -> Snapshot:
        constant_orm = await ConstantsORM.get(constants_id)
        snapshot_orm = SnapshotORM(
            step=snapshot.step,
            constants=constant_orm,
            particles=snapshot.particles,
            batch_id=snapshot.batch_id,
        )
        await snapshot_orm.insert()
        return build_snapshot(snapshot_orm)

    async def persist(self, snapshot: Snapshot):
        constant_orm = await ConstantsORM.get(snapshot.constants.id)
        snapshot_orm = SnapshotORM(
            step=snapshot.step,
            constants=constant_orm,
            particles=snapshot.particles,
            batch_id=snapshot.batch_id,
        )
        await snapshot_orm.insert()
        return build_snapshot(snapshot_orm)
