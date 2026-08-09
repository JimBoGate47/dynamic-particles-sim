from bson import ObjectId
from pydantic import BaseModel

from config.database import db_connection
from src.common.domain.entities.particle import Particle
from src.common.domain.enums import ConfinementType
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.particle_system_2d_builder import ParticleSystem2DBuilder
from src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from src.simulator.applitacion.use_cases.snapshot_finder import SnapshotFinderById
from src.simulator.domain.constants import DeviceType
from src.simulator.applitacion.use_cases.snapshot_lister import SnapshotsLister
from src.simulator.applitacion.use_cases.simulation_plus_gravity_runner import SimulationPlusGravityRunner
from src.simulator.applitacion.use_cases.simulation_runner import SimulationStabilizerRunner



class GetSnapshotRequest(BaseModel):
    snapshot_id: str


class ListSnapshotsRequest(BaseModel):
    constants_name: str


class ParticlePayload(BaseModel):
    r: list[float]
    v: list[float]
    a: list[float]
    phys_props: dict = {}


class CreateSnapshotRequest(BaseModel):
    step: int
    constants_id: str
    particles: list[ParticlePayload] = []
    batch_id: str | None = None
    n_particles: int = 32
    R: float = 6.0
    device: str = "cpu"


class RunSimulationRequest(BaseModel):
    snapshot_id: str
    n_steps: int = 506
    save_at_mod: int = 100
    wall: ConfinementType = ConfinementType.HARMONIC


class RunSimulationWithGravityRequest(BaseModel):
    snapshot_id: str
    stabilization_steps: int = 506
    n_steps: int = 10
    save_at_mod: int = 100
    wall: ConfinementType = ConfinementType.HARMONIC


async def get_snapshot(req: GetSnapshotRequest) -> dict | None:
    async with db_connection():
        snapshot = await SnapshotFinderById(
            snapshot_id=req.snapshot_id,
            orm_snapshot=ORMSnapshotRepository(),
        ).execute()
        return snapshot.model_dump(mode="json")


async def list_snapshots(req: ListSnapshotsRequest) -> list[dict]:
    async with db_connection():
        collections = await SnapshotsLister(
            filters=SnapshotsFilter(
                constants_name=req.constants_name,
            ),
            snapshot_repository=ORMSnapshotRepository(),
        ).execute()
        return [col.model_dump(mode="json") for col in collections]


async def create_snapshot(req: CreateSnapshotRequest) -> dict:
    if not req.particles:
        device = DeviceType(req.device)
        ps_domain = await ParticleSystem2DBuilder(
            n_particles=req.n_particles,
            R=req.R,
            device=device,
        ).execute()
        particles = ps_domain.particles
    else:
        particles = [Particle(**p.model_dump()) for p in req.particles]

    async with db_connection():
        snapshot = await SnapshotBuilder(
            step=req.step,
            constants_id=ObjectId(req.constants_id),
            particles=particles,
            batch_id=req.batch_id,
            orm_snapshot=ORMSnapshotRepository(),
            orm_constants=ORMConstantsRepository(),
        ).execute()
        return snapshot.model_dump(mode="json")


async def run_simulation(req: RunSimulationRequest) -> list[dict]:
    async with db_connection():
        snapshots = await SimulationStabilizerRunner(
            snapshot_id=req.snapshot_id,
            orm_snapshot=ORMSnapshotRepository(),
            fetch_links=True,
            n_steps=req.n_steps,
            save_at_mod=req.save_at_mod,
            wall=req.wall,
        ).execute()
        return [s.model_dump(mode="json") for s in snapshots]


async def run_simulation_with_gravity(req: RunSimulationWithGravityRequest) -> list[dict]:
    async with db_connection():
        snapshots = await SimulationPlusGravityRunner(
            snapshot_id=req.snapshot_id,
            orm_snapshot=ORMSnapshotRepository(),
            fetch_links=True,
            stabilization_steps=req.stabilization_steps,
            n_steps=req.n_steps,
            wall=req.wall,
        ).execute()
        return [s.model_dump(mode="json") for s in snapshots]
