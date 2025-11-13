from dataclasses import dataclass

from backend.src.common.domain.entities import Particle


# TODO tal vez no es necesario, igual a Snapshot?
@dataclass
class ParticleSystem2D:
    particles: list[Particle]

    def to_dict(self) -> dict:
        return {
            "particles": [
                particle.to_json()
                for particle in self.particles
            ]
        }
