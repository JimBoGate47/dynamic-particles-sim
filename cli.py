import asyncio

import torch

from backend.src.common.domain.entities.particle_system import ParticleSystem2D
from backend.src.common.domain.entities.properties import PhysicalProps, SimulationProps
from backend.src.common.infrastructure.repositories.constants import ORMConstantsRepository
from backend.src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from backend.src.simulator.applitacion.use_cases.constants_builder import ConstantsBuilder
from backend.src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from backend.src.simulator.applitacion.use_cases.velocity_verlet_applier import VelocityVerletApplier
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from backend.src.simulator.infrastructure.builders.particle_system import build_particles_2d
from backend.src.simulator.infrastructure.interaction import (
    PairElectrostaticInteraction,
    PotencialWallInteractionDecorator,
    FrictionInteractionDecorator,
)
from config.database import db_connection

interactions = PairElectrostaticInteraction()
interactions = PotencialWallInteractionDecorator(interactions)
interactions_plus_friction = FrictionInteractionDecorator(interactions)


async def main():
    RADIO = 6.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ps = ParticleSystem2DTensor(
        pos=ParticleSystem2DTensor.initialize_particles_in_circle(
            n_particles=6,
            R=1,
            device=device,
        ),
        sim_props=SimulationProps(
            g=9,
            k=10,
            min_vel=0,
            r_confinement=RADIO,
            k_confinement=0.5,
            beta=0.8,
            dt=0.1,
        ),
        phys_props=PhysicalProps.from_charges(
            n_particles=N_PARTICLES,
            charges=[1.0],
            device=device,
        )
    )
    async with db_connection():
        # TODO construir a partir de SimulationProps
        constants = await ConstantsBuilder(
            name="nombre3",
            g=ps.sim_props.g,
            k=ps.sim_props.k,
            dt=ps.sim_props.dt,
            min_vel=ps.sim_props.min_vel,
            orm_constants=ORMConstantsRepository(),
        ).execute()

        for _ in range(506):
            particles_system: ParticleSystem2D = build_particles_2d(ps)
            if ps.step in [100, 200, 300, 400, 500]:
                snap = await SnapshotBuilder(
                    step=ps.step,
                    constants_id=constants.id_object,
                    particles=particles_system.particles,
                    orm_snapshot=ORMSnapshotRepository(),
                    orm_constants=ORMConstantsRepository(),
                ).execute()
                print(snap)
            await VelocityVerletApplier(
                particle_system=ps,
                interactions=interactions,
            ).execute()


asyncio.run(main())
