from backend.src.common.domain.entities import Particle
from backend.src.common.domain.entities.properties import PhysicalProps
from backend.src.common.domain.types.properties import PhysicalProperties
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2D


def build_particles(
        mx_particles: ParticleSystem2D,
        phys_props: PhysicalProps,

) -> list[Particle]:
    particles = []
    for r, v, a in zip(mx_particles.pos, mx_particles.vel, mx_particles.acc):
        # obj_particles.append(
        #     Particle2D(
        #         r=Position2D(x=r[0].item(), y=r[1].item()),
        #         v=Velocity2D(x=v[0].item(), y=v[1].item()),
        #         a=Aceleration2D(x=a[0].item(), y=a[1].item()),
        #         phys_props=PhysicalProperties(q=1, m=1),
        #     )
        # )
        particles.append(
            Particle(
                r=[r[0].item(), r[1].item()],
                v=[v[0].item(), v[1].item()],
                a=[a[0].item(), a[1].item()],
                phys_props=phys_props.model_dump()
            )
        )
    return particles
