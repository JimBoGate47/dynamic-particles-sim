import asyncio
from pprint import pprint

import torch

from backend.src.common.domain.entities.particle_system import ParticleSystem2D
from backend.src.common.domain.entities.properties import PhysicalProps, SimulationProps
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
    N_PARTICLES = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ps = ParticleSystem2DTensor(
        pos=ParticleSystem2DTensor.initialize_particles_in_circle(
            n_particles=N_PARTICLES,
            R=4,
            device=device,
        ),
        sim_props=SimulationProps(
            g=9,
            k=2,
            min_vel=0,
            r_confinement=RADIO,
            k_confinement=0.5,
            beta=0.8,
            dt=0.1,
        ),
        phys_props=PhysicalProps.from_charges(
            n_particles=N_PARTICLES,
            charges=[10.0],
            device=device,
        )
    )
    async with db_connection():
        for _ in range(3):
            await VelocityVerletApplier(
                particle_system=ps,
                interactions=interactions,
            ).execute()
            particles_system: ParticleSystem2D = build_particles_2d(ps)
            pprint(particles_system.to_dict())


asyncio.run(main())
