from dataclasses import dataclass

from src.common.domain.entities import Constants
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.constants import ConstantsRepository


@dataclass
class ConstantsBuilder(UseCase):
    orm_constants: ConstantsRepository
    name: str
    g: float
    k: float
    dt: float
    min_vel: float
    friction: float = 0
    confinement: str = "radial"
    r_confinement: float = 0
    ruta: bool = False
    version: str = "v1"
    barra_height: float = 0
    barra_qlamb: float = 0

    async def execute(self, *args, **kwargs) -> Constants:
        constants = Constants(
            name=self.name,
            g=self.g,
            k=self.k,
            dt=self.dt,
            min_vel=self.min_vel,
            friction=self.friction,
            confinement=self.confinement,
            r_confinement=self.r_confinement,
            ruta=self.ruta,
            version=self.version,
            barra_height=self.barra_height,
            barra_qlamb=self.barra_qlamb,
        )
        return await self.orm_constants.persist(constants)
