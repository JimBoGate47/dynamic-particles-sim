from beanie import Document, Indexed

from src.common.domain.entities.properties import SimulationProps


class ConstantsORM(Document):
    name: Indexed(str, unique=True)
    sim_props: SimulationProps
    friction: float = 0
    confinement: str = "radial"
    ruta: bool = False
    version: str = "v1"
    barra_height: float = 0
    barra_qlamb: float = 0
