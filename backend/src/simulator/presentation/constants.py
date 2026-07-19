from config.database import db_connection
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.simulator.applitacion.use_cases.constants_finder import ConstantsFinder


async def find_constants() -> list[dict]:
    async with db_connection():
        # TODO construir a partir de SimulationProps
        finder = ConstantsFinder(
            orm_constants=ORMConstantsRepository(),
        )
        constants = await finder.execute()
        return [
            constant.model_dump(mode="json")
            for constant in constants
        ]
