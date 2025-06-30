import asyncio

import torch

from backend.src.common.domain.types.properties import SimulationProperties, PhysicalProperties
from backend.src.common.infrastructure.repositories.constants import ORMConstantsRepository
from backend.src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from backend.src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2D
from backend.src.simulator.infrastructure.builders.constants import build_constants
from backend.src.simulator.infrastructure.builders.particle import build_particles
from backend.src.simulator.infrastructure.builders.snapshot import build_snapshot
from config.utils import connect, disconnect


async def main():
    RADIO = 6.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ps = ParticleSystem2D(
        pos=ParticleSystem2D.initialize_particles_in_circle(
            n_particles=6,
            R=1,
            device=device,
        ),
        sim_props=SimulationProperties(
            g=9,
            k=10,
            min_vel=0,
            r_confinement=RADIO,
            beta=0.5,
            dt=0.1,
        ),
        phys_props=PhysicalProperties(
            q=1.0,
            m=1.0,
        )
    )
    constants = build_constants(sim_props=ps.sim_props)
    particles = build_particles(ps, phys_props=ps.phys_props)
    # snapshot = build_snapshot(ps)

    await connect()
    snap = await SnapshotBuilder(
        step=0,
        constants=constants,
        constants_id=None,
        particles=particles,
        orm_snapshot=ORMSnapshotRepository(),
        orm_constants=ORMConstantsRepository(),
    ).execute()
    print(snap)

    await disconnect()


asyncio.run(main())
