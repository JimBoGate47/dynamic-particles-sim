from pydantic import BaseModel

from bson import ObjectId

from src.common.domain.entities.properties import SimulationProps
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.constants_builder import ConstantsBuilder
from src.simulator.applitacion.use_cases.constants_deleter import ConstantsDeleter
from src.simulator.applitacion.use_cases.constants_finder import ConstantsFinder
from config.database import db_connection


class CreateConstantsRequest(BaseModel):
    name: str
    sim_props: SimulationProps
    friction: float = 0
    confinement: str = "radial"
    ruta: bool = False
    version: str = "v1"
    barra_height: float = 0
    barra_qlamb: float = 0


class DeleteConstantsRequest(BaseModel):
    constants_id: str | None = None
    name: str | None = None


async def find_constants() -> list[dict]:
    async with db_connection():
        finder = ConstantsFinder(
            orm_constants=ORMConstantsRepository(),
        )
        constants = await finder.execute()
        return [
            constant.model_dump(mode="json")
            for constant in constants
        ]


async def create_constants(req: CreateConstantsRequest) -> dict:
    async with db_connection():
        constants = await ConstantsBuilder(
            orm_constants=ORMConstantsRepository(),
            name=req.name,
            sim_props=req.sim_props,
            friction=req.friction,
            confinement=req.confinement,
            ruta=req.ruta,
            version=req.version,
            barra_height=req.barra_height,
            barra_qlamb=req.barra_qlamb,
        ).execute()
        return constants.model_dump(mode="json")


async def delete_constants(req: DeleteConstantsRequest) -> dict:
    async with db_connection():
        deleted = await ConstantsDeleter(
            orm_constants=ORMConstantsRepository(),
            orm_snapshot=ORMSnapshotRepository(),
            constants_id=ObjectId(req.constants_id) if req.constants_id else None,
            name=req.name,
        ).execute()
        return {"deleted": deleted}
