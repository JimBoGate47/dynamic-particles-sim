from bson import ObjectId

from src.common.domain.entities import Constants
from src.common.domain.models.constants import ConstantsORM
from src.common.domain.repositories.constants import ConstantsRepository
from src.common.infrastructure.builders.constants import build_constants


class ORMConstantsRepository(ConstantsRepository):
    async def find_all(self):
        constants = await ConstantsORM.find().to_list()
        return [
            build_constants(constant)
            for constant in constants
        ]

    async def find_by_task_name(self, name: str):
        constants = await ConstantsORM.find(ConstantsORM.name == name).to_list()
        return [
            build_constants(constant)
            for constant in constants
        ]

    async def find_by_id(self, _id: ObjectId) -> Constants | None:
        orm_instance = await ConstantsORM.find_one(ConstantsORM.id == _id)
        if not orm_instance:
            return None
        return build_constants(orm_instance)

    async def persist(self, constant: Constants) -> Constants:
        constants = ConstantsORM(
            name=constant.name,
            sim_props=constant.sim_props,
            friction=constant.friction,
            confinement=constant.confinement,
            ruta=constant.ruta,
            version=constant.version,
            barra_height=constant.barra_height,
            barra_qlamb=constant.barra_qlamb,
        )
        await constants.insert()
        return build_constants(constants)
