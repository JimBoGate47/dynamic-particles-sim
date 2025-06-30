from typing import Optional

from pydantic import BaseModel


class Constants(BaseModel):
    id: Optional[str] = None
    name: str
    g: float
    k: float
    dt: float
    min_vel: float = 0
    friction: float = 0
    confinement: str = "radial"
    r_confinement: float = 0
    ruta: bool = False
    version: str = "v1"
    barra_height: float = 0
    barra_qlamb: float = 0

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "g": self.g,
            "k": self.k,
            "dt": self.dt,
            "min_vel": self.min_vel,
            "friction": self.friction,
            "confinement": self.confinement,
            "r_confinement": self.r_confinement,
            "ruta": self.ruta,
            "version": self.version,
            "barra_height": self.barra_height,
            "barra_qlamb": self.barra_qlamb,
        }
