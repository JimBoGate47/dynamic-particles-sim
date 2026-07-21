import asyncio
import uuid

from config.database import db_connection
from src.common.domain.entities.particle_system import ParticleSystem2D
from src.common.domain.entities.properties import SimulationProps
from src.common.infrastructure.repositories.constants import ORMConstantsRepository
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.constants_builder import ConstantsBuilder
from src.simulator.applitacion.use_cases.particle_system_2d_builder import ParticleSystem2DBuilder
from src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from src.simulator.applitacion.use_cases.velocity_verlet_applier import VelocityVerletApplier
from src.simulator.domain.constants import DeviceType
from src.simulator.infrastructure.builders.particle_system import build_particles_2d, build_particles_2d_tensor
from src.simulator.infrastructure.interaction import (
    PairElectrostaticInteraction,
    PotencialWallInteractionDecorator,
    FrictionInteractionDecorator,
)

interactions = PairElectrostaticInteraction()
interactions = PotencialWallInteractionDecorator(interactions)
interactions_plus_friction = FrictionInteractionDecorator(interactions)


async def main():
    RADIO = 6.0
    N_PARTICLES = 32
    sim_props = SimulationProps(
        g=9,
        k=10,
        min_vel=0,
        r_confinement=RADIO,
        k_confinement=0.5,
        beta=0.6,
        dt=0.1,
    )
    ps_domain = await ParticleSystem2DBuilder(
        n_particles=N_PARTICLES,
        R=RADIO,
        device=DeviceType.CPU,
    ).execute()
    ps = build_particles_2d_tensor(ps_domain, device=DeviceType.CPU)
    async with db_connection():
        constants = await ConstantsBuilder(
            name="nombre3",
            sim_props=sim_props,
            orm_constants=ORMConstantsRepository(),
        ).execute()

        batch_id = str(uuid.uuid4())
        for _ in range(506):
            particles_system: ParticleSystem2D = build_particles_2d(ps)
            if ps.step in [100, 200, 300, 400, 500]:
                snap = await SnapshotBuilder(
                    step=ps.step,
                    constants_id=constants.id_object,
                    particles=particles_system.particles,
                    batch_id=batch_id,
                    orm_snapshot=ORMSnapshotRepository(),
                    orm_constants=ORMConstantsRepository(),
                ).execute()
                print(snap)
            await VelocityVerletApplier(
                particle_system=ps,
                sim_props=sim_props,
                interactions=interactions_plus_friction,
            ).execute()


asyncio.run(main())
