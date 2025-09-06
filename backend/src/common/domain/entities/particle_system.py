from dataclasses import dataclass

from backend.src.common.domain.entities import Particle

# TODO tal vez no es necesario, igual a Snapshot?
@dataclass
class ParticleSystem2D:
    particles: list[Particle]

