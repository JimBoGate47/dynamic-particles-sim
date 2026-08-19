import asyncio

from bson import ObjectId

from src.common.domain.entities.constants import Constants
from src.common.domain.entities.properties import SimulationProps
from src.simulator.applitacion.use_cases.constants_deleter import ConstantsDeleter


class FakeConstantsRepository:
    def __init__(self, constants: list[Constants]):
        self._constants = constants
        self._deleted_ids: list[ObjectId] = []
        self._deleted_names: list[str] = []

    async def find_by_id(self, _id):
        for c in self._constants:
            if c.id == str(_id):
                return c
        return None

    async def find_by_task_name(self, name):
        return [c for c in self._constants if c.name == name]

    async def delete_by_id(self, _id):
        self._deleted_ids.append(_id)
        return any(c.id == str(_id) for c in self._constants)

    async def delete_by_name(self, name):
        self._deleted_names.append(name)
        return any(c.name == name for c in self._constants)


class FakeSnapshotRepository:
    def __init__(self):
        self.deleted_constants_ids: list[ObjectId] = []

    async def delete_with_constants_id(self, _id):
        self.deleted_constants_ids.append(_id)


def _run(coro):
    return asyncio.run(coro)


def _constants(_id: str, name: str = "sim") -> Constants:
    return Constants(
        id=_id,
        name=name,
        sim_props=SimulationProps(
            g=9.0,
            k=10.0,
            min_vel=0.0,
            r_confinement=6.0,
            k_confinement=0.5,
            beta=0.6,
            dt=0.1,
        ),
    )


class TestConstantsDeleter:
    def test_deletes_by_id_and_related_snapshots(self):
        c = _constants("507f1f77bcf86cd799439011", name="sim-a")
        constants_repo = FakeConstantsRepository([c])
        snapshots_repo = FakeSnapshotRepository()

        deleted = _run(ConstantsDeleter(
            orm_constants=constants_repo,
            orm_snapshot=snapshots_repo,
            constants_id=ObjectId(c.id),
        ).execute())

        assert deleted is True
        assert constants_repo._deleted_ids == [ObjectId(c.id)]
        assert snapshots_repo.deleted_constants_ids == [ObjectId(c.id)]

    def test_deletes_by_name_and_related_snapshots(self):
        c = _constants("507f1f77bcf86cd799439012", name="sim-b")
        constants_repo = FakeConstantsRepository([c])
        snapshots_repo = FakeSnapshotRepository()

        deleted = _run(ConstantsDeleter(
            orm_constants=constants_repo,
            orm_snapshot=snapshots_repo,
            name="sim-b",
        ).execute())

        assert deleted is True
        assert constants_repo._deleted_names == ["sim-b"]
        assert snapshots_repo.deleted_constants_ids == [ObjectId(c.id)]

    def test_returns_false_when_id_not_found(self):
        constants_repo = FakeConstantsRepository([
            _constants("507f1f77bcf86cd799439013"),
        ])
        snapshots_repo = FakeSnapshotRepository()

        deleted = _run(ConstantsDeleter(
            orm_constants=constants_repo,
            orm_snapshot=snapshots_repo,
            constants_id=ObjectId("507f1f77bcf86cd799439099"),
        ).execute())

        assert deleted is False
        assert snapshots_repo.deleted_constants_ids == []

    def test_returns_false_without_identifier(self):
        constants_repo = FakeConstantsRepository([])
        snapshots_repo = FakeSnapshotRepository()

        deleted = _run(ConstantsDeleter(
            orm_constants=constants_repo,
            orm_snapshot=snapshots_repo,
        ).execute())

        assert deleted is False