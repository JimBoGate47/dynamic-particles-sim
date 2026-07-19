from abc import abstractmethod, ABC
from typing import List

from bson import ObjectId

from src.common.domain.entities import Constants


class ConstantsRepository(ABC):
    @abstractmethod
    async def find_all(self) -> List[Constants]:
        raise NotImplementedError

    @abstractmethod
    async def find_by_task_name(self, name: str) -> List[Constants]:
        raise NotImplementedError

    @classmethod
    async def find_by_id(self, _id: ObjectId) -> Constants | None:
        raise NotImplementedError

    @abstractmethod
    async def persist(self, constant: Constants) -> Constants:
        raise NotImplementedError
