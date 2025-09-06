from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel

from backend.src.common.domain.entities.base import CustomBaseModel
from backend.src.common.domain.entities.particle import Particle
from backend.src.common.domain.entities.constants import Constants


class Snapshot(CustomBaseModel):
    id: str | None = None
    step: int
    constants: Optional[Constants]
    particles: List[Particle]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "constants": self.constants.to_json() if self.constants else None,
            "particles": [particle.to_json() for particle in self.particles],
        }

    def to_plain_dict(self) -> dict:
        return {
            "id": self.id,
            "constants": self.constants.to_json() if self.constants else None,
            "particles": self.export_particles(),
        }

    def export_particles(self) -> List[dict]:
        return [
            {
                "step": self.step,
                "p_idx": idx,
                "rx": particle.r[0],
                "ry": particle.r[1],
                "charge": particle.phys_props["q"],
            }
            for idx, particle in enumerate(self.particles)
        ]
