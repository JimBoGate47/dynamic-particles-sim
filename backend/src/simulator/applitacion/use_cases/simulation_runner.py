from dataclasses import dataclass, field

from loguru import logger

from src.common.domain.entities.particle_system import ParticleSystem2D
from src.common.domain.interfaces import UseCase
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.mixins import SnapshotFinderMixin
from src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from src.simulator.applitacion.use_cases.velocity_verlet_applier import VelocityVerletApplier
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.domain.interfaces import Interaction
from src.simulator.infrastructure.builders.particle_system import build_particles_2d
from src.simulator.infrastructure.builders.system_tensor import build_system_tensor


@dataclass
class SimulationRunner(UseCase, SnapshotFinderMixin):
    interactions: Interaction
    n_steps: int = 506
    record_steps: set[int] = field(default_factory=lambda: {100, 200, 300, 400, 500})
    fetch_links: bool

    async def execute(self, *args, **kwargs) -> list[dict]:
        snapshot = await self.find_by_id()
        if not snapshot:
            raise ValueError("Snapshot not found")
        if not snapshot.constants:
            raise ValueError("Snapshot has no linked constants")

        system_tensor = build_system_tensor(snapshot)
        ps = ParticleSystem2DTensor(
            pos=system_tensor.pos,
            vel=system_tensor.vel,
            acc=system_tensor.acc,
            phys_props=system_tensor.phys_props,
        )

        log_interval = max(1, self.n_steps // 10)
        logger.info("Starting simulation: step={} batch_id={} sim_props={}",
                     ps.step, snapshot.batch_id, snapshot.constants.sim_props)
        snapshots = []
        for _ in range(self.n_steps):
            particles_system: ParticleSystem2D = build_particles_2d(ps)
            if ps.step % log_interval == 0:
                logger.debug("Simulation progress: step={}", ps.step)
            if ps.step in self.record_steps:
                snap = await SnapshotBuilder(
                    step=ps.step,
                    constants_id=snapshot.constants.id_object,
                    particles=particles_system.particles,
                    batch_id=snapshot.batch_id,
                    orm_snapshot=ORMSnapshotRepository(),
                    orm_constants=ORMConstantsRepository(),
                ).execute()
                snapshots.append(snap)
            await VelocityVerletApplier(
                particle_system=ps,
                sim_props=snapshot.constants.sim_props,
                interactions=self.interactions,
            ).execute()
        return snapshots
