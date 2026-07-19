from dataclasses import dataclass
from typing import Optional, List

from src.common.domain.entities import Constants
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.constants import ConstantsRepository


@dataclass
class ConstantsFinder(UseCase):
    orm_constants: ConstantsRepository
    name: Optional[str] = None

    async def execute(self, *args, **kwargs) -> List[Constants]:
        if self.name:
            return await self.orm_constants.find_by_task_name(name=self.name)
        return await self.orm_constants.find_all()
