from src.common.domain.entities.properties import SimulationProps
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.simulator.applitacion.use_cases.constants_builder import ConstantsBuilder
from src.simulator.applitacion.use_cases.constants_finder import ConstantsFinder
from config.database import db_connection


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
        raw_sim = data.get("sim_props", data)
        sim_props = SimulationProps(
            g=raw_sim["g"],
            k=raw_sim["k"],
            dt=raw_sim["dt"],
            min_vel=raw_sim.get("min_vel", 0),
            r_confinement=raw_sim.get("r_confinement", 0),
            k_confinement=raw_sim.get("k_confinement", 0),
            beta=raw_sim.get("beta", 0),
        )
        constants = await ConstantsBuilder(
            orm_constants=ORMConstantsRepository(),
            name=data["name"],
            sim_props=sim_props,
            friction=data.get("friction", 0),
            confinement=data.get("confinement", "radial"),
            ruta=data.get("ruta", False),
            version=data.get("version", "v1"),
            barra_height=data.get("barra_height", 0),
            barra_qlamb=data.get("barra_qlamb", 0),
        ).execute()
        return constants.model_dump(mode="json")
