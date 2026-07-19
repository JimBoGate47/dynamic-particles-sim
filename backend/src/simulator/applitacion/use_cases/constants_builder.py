from dataclasses import dataclass

from src.common.domain.entities import Constants
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.constants import ConstantsRepository


@dataclass
class ConstantsBuilder(UseCase):
    name: str
    g: float
    k: float
    dt: float
    min_vel: float
    orm_constants: ConstantsRepository

    async def execute(self, *args, **kwargs) -> Constants:
        constants = Constants(
            name=self.name,
            g=self.g,
            k=self.k,
            dt=self.dt,
            min_vel=self.min_vel,
        )
        return await self.orm_constants.persist(constants)
