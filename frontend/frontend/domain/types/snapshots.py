from pydantic import BaseModel, computed_field

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
    batch_id: str


class SnapshotsCollection(BaseModel):
    batch_id: str
    snapshots: list[Snapshot]

    @computed_field
    @property
    def steps(self) -> list[int]:
        return [s.step for s in self.snapshots]
