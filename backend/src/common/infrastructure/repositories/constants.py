from backend.src.common.domain.models.constants import ConstantsORM
from backend.src.common.domain.entities import Constants
from backend.src.common.infrastructure.builders.constants import build_constants
from backend.src.common.domain.repositories.constants import ConstantsRepository


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

    async def find_by_id(self, _id: str):
        constants = await ConstantsORM.find_one(ConstantsORM.id == _id)
        return [
            build_constants(constant)
            for constant in constants
        ]

    async def persist(self, constant: Constants) -> Constants:
        constants = ConstantsORM(
            name=constant.name,
            g=constant.g,
            k=constant.k,
            min_vel=constant.min_vel,
            friction=constant.friction,
            ruta=constant.ruta,
            dt=constant.dt,
            confinement=constant.confinement,
            r_confinment=constant.r_confinement,
            version=constant.version,
            barra_height=constant.barra_height,
            barra_qlamb=constant.barra_qlamb,
        )
        print("JIM")
        print(constants)
        await constants.insert()
        return build_constants(constants)
