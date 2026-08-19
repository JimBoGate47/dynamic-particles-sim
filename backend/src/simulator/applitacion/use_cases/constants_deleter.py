from dataclasses import dataclass
from typing import Optional

from bson import ObjectId

from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.constants import ConstantsRepository
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class ConstantsDeleter(UseCase):
    orm_constants: ConstantsRepository
    orm_snapshot: SnapshotRepository
    constants_id: Optional[ObjectId] = None
    name: Optional[str] = None

    async def execute(self, *args, **kwargs) -> bool:
        if self.constants_id is not None:
            constant = await self.orm_constants.find_by_id(self.constants_id)
            if constant is None:
                return False
            constants_id = ObjectId(constant.id)
            deleted_constants = await self.orm_constants.delete_by_id(constants_id)
        elif self.name is not None:
            constants = await self.orm_constants.find_by_task_name(self.name)
            if not constants:
                return False
            constants_id = ObjectId(constants[0].id)
            deleted_constants = await self.orm_constants.delete_by_name(self.name)
        else:
            return False

        if deleted_constants:
            await self.orm_snapshot.delete_with_constants_id(constants_id)

        return deleted_constants