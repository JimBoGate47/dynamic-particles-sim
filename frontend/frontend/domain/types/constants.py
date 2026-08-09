from pydantic import BaseModel, model_validator


class SimulationProps(BaseModel):
    g: float
    k: float
    dt: float
    min_vel: float = 0
    r_confinement: float = 0
    k_confinement: float = 0
    beta: float = 0
    delta_gravity: float | None = None


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

    @model_validator(mode="after")
    def _name_not_empty(self):
        if not self.name:
            raise ValueError("name no puede estar vacío")
        return self
