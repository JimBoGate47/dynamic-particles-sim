import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field

from src.common.domain.entities.base import CustomBaseModel
from src.common.domain.entities.constants import Constants
from src.common.domain.entities.particle import Particle


class Snapshot(CustomBaseModel):
    id: str | None = None
    step: int
    constants: Optional[Constants]
    particles: List[Particle]
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "constants": self.constants.to_json() if self.constants else None,
            "particles": [particle.to_json() for particle in self.particles],
            "batch_id": self.batch_id,
        }

    def to_plain_dict(self) -> dict:
        return {
            "id": self.id,
            "constants": self.constants.to_json() if self.constants else None,
            "particles": self.export_particles(),
            "batch_id": self.batch_id,
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


class SnapshotsCollection(BaseModel):
    batch_id: str
    snapshots: list[Snapshot]

    @computed_field
    @property
    def steps(self) -> list[int]:
        return [s.step for s in self.snapshots]
