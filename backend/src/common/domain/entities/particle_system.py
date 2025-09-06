from dataclasses import dataclass

from backend.src.common.domain.entities import Particle


@dataclass
class ParticleSystem2D:
    particles: list[Particle]

