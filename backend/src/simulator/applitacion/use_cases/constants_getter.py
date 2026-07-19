from dataclasses import dataclass

from bson import ObjectId

from src.common.domain.entities import Constants
from src.common.domain.repositories.constants import ConstantsRepository
from src.common.domain.interfaces import UseCase


@dataclass
class ConstantsGetter(UseCase):
    constants_id: ObjectId
    orm_constants: ConstantsRepository

    async def execute(self, *args, **kwargs) -> Constants:
        return await self.orm_constants.find_by_id(self.constants_id)
