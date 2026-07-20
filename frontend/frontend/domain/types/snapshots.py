from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

from frontend.domain.types.constants import Constants


class Particle(BaseModel):
    r: list[float]
    v: list[float]
    a: list[float]
    phys_props: dict = {}


class Snapshot(BaseModel):
    id: str
    step: int
    constants: Constants
    particles: list[Particle]


class SnapshotsCollection(BaseModel):
    meta_id: str = Field(default_factory=lambda: str(uuid4()))
    snapshots: list[Snapshot]

    @computed_field
    @property
    def steps(self) -> list[int]:
        return [s.step for s in self.snapshots]
