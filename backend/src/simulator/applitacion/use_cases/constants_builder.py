from dataclasses import dataclass

from backend.src.common.domain.entities import Constants
from backend.src.common.domain.repositories.constants import ConstantsRepository
from backend.src.common.domain.interfaces import UseCase


@dataclass
class ConstantsBuilder(UseCase):
    constants: Constants
    orm_constants: ConstantsRepository

    async def execute(self, *args, **kwargs) -> Constants:
        return await self.orm_constants.persist(self.constants)
