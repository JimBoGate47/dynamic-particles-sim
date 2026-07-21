import torch

from src.common.domain.entities.particle import Particle
from src.common.domain.entities.particle_system import ParticleSystem2D
from src.common.domain.entities.properties import PhysicalProps
from src.simulator.domain.constants import DeviceType, DEVICE_MAP
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


def build_particles_2d_tensor(
        particle_system: ParticleSystem2D,
        device: DeviceType = DeviceType.AUTO,
) -> ParticleSystem2DTensor:
    device = DEVICE_MAP[device]

    particles = particle_system.particles
    n = len(particles)

    pos = torch.zeros((n, 2), device=device)
    vel = torch.zeros((n, 2), device=device)
    acc = torch.zeros((n, 2), device=device)
    q = torch.zeros((n, 1), device=device)
    m = torch.zeros((n, 1), device=device)

    for i, p in enumerate(particles):
        pos[i] = torch.tensor([p.r[0], p.r[1]], dtype=torch.float32, device=device)
        vel[i] = torch.tensor([p.v[0], p.v[1]], dtype=torch.float32, device=device)
        acc[i] = torch.tensor([p.a[0], p.a[1]], dtype=torch.float32, device=device)
        q[i] = float(p.phys_props["q"])
        m[i] = float(p.phys_props["m"])

    phys_props = PhysicalProps(q=q, m=m)

    return ParticleSystem2DTensor(
        pos=pos,
        vel=vel,
        acc=acc,
        phys_props=phys_props,
    )
