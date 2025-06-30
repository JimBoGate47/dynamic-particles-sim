from dataclasses import dataclass
from typing import overload

from backend.src.common.domain.types.properties import PhysicalProperties


@dataclass
class Data2D:
    x: float
    y: float


class Position2D(Data2D):
    pass


class Velocity2D(Data2D):
    pass


class Aceleration2D(Data2D):
    pass


@dataclass
class Particle:
    r: Data2D
    v: Data2D
    a: Data2D


@dataclass
class Particle2D(Particle):
    r: Position2D
    v: Velocity2D
    a: Aceleration2D
    phys_props: PhysicalProperties
