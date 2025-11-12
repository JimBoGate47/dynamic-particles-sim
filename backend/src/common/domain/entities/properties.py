from pydantic import BaseModel


# TODO tal vez como id en lugar de UUID?
class SimulationID(BaseModel):
    id: float
    name: float


class PhysicalProps(BaseModel):
    q: float
    m: float


class SimulationProps(BaseModel):
    g: float
    k: float
    min_vel: float
    r_confinement: float
    k_confinement: float
    beta: float
    dt: float
