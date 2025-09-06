import asyncio

import torch

from backend.src.common.domain.entities.properties import PhysicalProps, SimulationProps
from backend.src.common.infrastructure.repositories.constants import ORMConstantsRepository
from backend.src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from backend.src.simulator.applitacion.use_cases.constants_builder import ConstantsBuilder
from backend.src.simulator.applitacion.use_cases.snapshot_builder import SnapshotBuilder
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2D
from backend.src.simulator.infrastructure.builders.particle import build_particles
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
        sim_props=SimulationProps(
            g=9,
            k=10,
            min_vel=0,
            r_confinement=RADIO,
            beta=0.5,
            dt=0.1,
        ),
        phys_props=PhysicalProps(
            q=1.0,
            m=1.0,
        )
    )
    await connect()
    constants = await ConstantsBuilder(
        name="nombre2",
        g=ps.sim_props.g,
        k=ps.sim_props.k,
        dt=ps.sim_props.dt,
        min_vel=ps.sim_props.min_vel,
        orm_constants=ORMConstantsRepository(),
    ).execute()
    particles = build_particles(ps, phys_props=ps.phys_props)

    snap = await SnapshotBuilder(
        step=0,
        constants_id=constants.id_object,
        particles=particles,
        orm_snapshot=ORMSnapshotRepository(),
        orm_constants=ORMConstantsRepository(),
    ).execute()
    print(snap)

    await disconnect()


asyncio.run(main())
