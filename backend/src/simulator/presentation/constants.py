from config.database import db_connection
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.simulator.applitacion.use_cases.constants_builder import ConstantsBuilder
from src.simulator.applitacion.use_cases.constants_finder import ConstantsFinder


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


async def create_constants(data: dict) -> dict:
    async with db_connection():
        constants = await ConstantsBuilder(
            orm_constants=ORMConstantsRepository(),
            name=data["name"],
            g=data["g"],
            k=data["k"],
            dt=data["dt"],
            min_vel=data.get("min_vel", 0),
            friction=data.get("friction", 0),
            confinement=data.get("confinement", "radial"),
            r_confinement=data.get("r_confinement", 0),
            ruta=data.get("ruta", False),
            version=data.get("version", "v1"),
            barra_height=data.get("barra_height", 0),
            barra_qlamb=data.get("barra_qlamb", 0),
        ).execute()
        return constants.model_dump(mode="json")
