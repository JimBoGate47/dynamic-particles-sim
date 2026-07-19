from src.common.domain.entities.particle import Particle
from src.common.domain.entities.particle_system import ParticleSystem2D
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor


def build_particles_2d(
        particles_tensor: ParticleSystem2DTensor,
) -> ParticleSystem2D:
    particles = []
    for r, v, a, q, m in zip(
            particles_tensor.pos,
            particles_tensor.vel,
            particles_tensor.acc,
            particles_tensor.phys_props.q,
            particles_tensor.phys_props.m,
    ):
        # obj_particles.append(
        #     Particle2D(
        #         r=Position2D(x=r[0].item(), y=r[1].item()),
        #         v=Velocity2D(x=v[0].item(), y=v[1].item()),
        #         a=Aceleration2D(x=a[0].item(), y=a[1].item()),
        #         phys_props=PhysicalProperties(q=1, m=1),
        #     )
        # )
        phys_props_dict = {
            "q": q.item(),
            "m": m.item(),
        }
        particles.append(
            Particle(
                r=[r[0].item(), r[1].item()],
                v=[v[0].item(), v[1].item()],
                a=[a[0].item(), a[1].item()],
                phys_props=phys_props_dict,
            )
        )
    return ParticleSystem2D(
        particles=particles,
    )
