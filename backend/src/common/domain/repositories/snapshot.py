from abc import abstractmethod, ABC
from typing import List, Optional

from src.common.domain.entities import Snapshot
from src.common.domain.entities.particle import Particle
from src.common.domain.filters.snapshot import SnapshotsFilter


class SnapshotRepository(ABC):
    @abstractmethod
    async def filter(self, params: SnapshotsFilter) -> list[Snapshot]:
        raise NotImplementedError

    @abstractmethod
    async def update_particles(self, _id, particles: List[Particle]) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def find_by_id(self, _id, fetch_links=False) -> Optional[Snapshot]:
        raise NotImplementedError

    @abstractmethod
    async def find_particles(self):
        raise NotImplementedError

    @abstractmethod
    async def find_with_constants_id(self, _id, fetch_links=False) -> Optional[List[Snapshot]]:
        raise NotImplementedError

    @abstractmethod
    async def delete_with_constants_id(self, _id):
        raise NotImplementedError

    @abstractmethod
    async def persist(self, snapshot: Snapshot):
        raise NotImplementedError

    @abstractmethod
    async def persist_with_constants_id(
            self,
            constants_id: str,
            snapshot: Snapshot,
            fetch_links: bool = False,
    ) -> Snapshot:
        raise NotImplementedError
