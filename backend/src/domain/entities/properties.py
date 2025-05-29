from dataclasses import dataclass


@dataclass
class PhysicalProperties:
    q: float
    m: float


@dataclass
class SimulationProperties:
    r_confin: float
    beta: float
    dt: float
