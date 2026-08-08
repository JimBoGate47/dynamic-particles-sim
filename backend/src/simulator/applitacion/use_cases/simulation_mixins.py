from dataclasses import dataclass

from bson import ObjectId
from loguru import logger

from src.common.domain.entities import Snapshot
from src.common.domain.entities.particle_system import ParticleSystem2D
from src.common.domain.entities.properties import SimulationProps
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from src.simulator.applitacion.use_cases.velocity_verlet_applier import VelocityVerletApplier
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.domain.interfaces import Interaction
from src.simulator.infrastructure.builders.particle_system import build_particles_2d


@dataclass
class SimulationStabilizerMixin:
    async def stabilize(
            self,
            ps: ParticleSystem2DTensor,
            constants_id: ObjectId,
            sim_props: SimulationProps,
            interactions: Interaction,
            stabilization_steps: int,
            save_at_mod: int,
            batch_id: str | None = None,
    ) -> list[Snapshot]:
        snapshots = []
        for n_step in range(1, stabilization_steps + 1):
            if save_at_mod > 0 and n_step % save_at_mod == 0:
                logger.debug("Simulation progress: step={}", ps.step)
                snapshot = await self.persist(
                    ps_tensor=ps,
                    constants_id=constants_id,
                    batch_id=batch_id,
                )
                snapshots.append(snapshot)
            await VelocityVerletApplier(
                particle_system=ps,
                sim_props=sim_props,
                interactions=interactions,
            ).execute()
        if save_at_mod == 0:
            logger.debug("Simulation progress: step={}", ps.step)
            snapshot = await self.persist(
                ps_tensor=ps,
                constants_id=constants_id,
                batch_id=batch_id,
            )
            snapshots.append(snapshot)
        return snapshots

    async def persist(
            self,
            ps_tensor: ParticleSystem2DTensor,
            constants_id: ObjectId,
            batch_id: str | None = None,
    ) -> Snapshot | None:
        particles_system: ParticleSystem2D = build_particles_2d(ps_tensor)
        return await SnapshotBuilder(
            step=ps_tensor.step,
            constants_id=constants_id,
            particles=particles_system.particles,
            batch_id=batch_id,
            orm_snapshot=ORMSnapshotRepository(),
            orm_constants=ORMConstantsRepository(),
        ).execute()
