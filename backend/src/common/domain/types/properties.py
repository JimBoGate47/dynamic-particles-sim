from dataclasses import dataclass


@dataclass
class PhysicalProperties:
    q: float
    m: float

    @property
    def to_dict(self) -> dict:
        return {
            "q": self.q,
            "m": self.m,
        }


@dataclass
class SimulationProperties:
    g: float
    k: float
    min_vel: float
    r_confinement: float
    beta: float
    dt: float

    @property
    def to_dict(self) -> dict:
        return {
            "r_confinement": self.r_confinement,
            "beta": self.beta,
            "dt": self.dt,
        }
