from src.common.domain.entities.base import CustomBaseModel
from src.common.domain.entities.properties import SimulationProps


class Constants(CustomBaseModel):
    id: str | None = None
    name: str
    sim_props: SimulationProps
    friction: float = 0
    confinement: str = "radial"
    ruta: bool = False
    version: str = "v1"
    barra_height: float = 0
    barra_qlamb: float = 0

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "g": self.sim_props.g,
            "k": self.sim_props.k,
            "dt": self.sim_props.dt,
            "min_vel": self.sim_props.min_vel,
            "friction": self.friction,
            "confinement": self.confinement,
            "r_confinement": self.sim_props.r_confinement,
            "ruta": self.ruta,
            "version": self.version,
            "barra_height": self.barra_height,
            "barra_qlamb": self.barra_qlamb,
        }
