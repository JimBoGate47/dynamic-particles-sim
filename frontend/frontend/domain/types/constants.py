from pydantic import BaseModel


class SimulationProps(BaseModel):
    g: float
    k: float
    dt: float
    min_vel: float = 0
    r_confinement: float = 0
    k_confinement: float = 0
    beta: float = 0


class Constants(BaseModel):
    id: str = ""
    name: str
    sim_props: SimulationProps
    friction: float = 0
    confinement: str
    ruta: bool
    version: str
    barra_height: float = 0
    barra_qlamb: float = 0
