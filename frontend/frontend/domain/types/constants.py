from pydantic import BaseModel


class Constants(BaseModel):
    id: str = ""
    name: str
    g: float
    k: float
    dt: float
    min_vel: float = 0
    friction: float = 0
    confinement: str
    r_confinement: float = 0
    ruta: bool
    version: str
    barra_height: float = 0
    barra_qlamb: float = 0
