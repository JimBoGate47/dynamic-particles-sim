import math
from dataclasses import dataclass
from typing import Optional

import torch

from backend.src.common.domain.entities.properties import SimulationProps, PhysicalProps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class System2DTensor:
    pos: torch.Tensor
    vel: Optional[torch.Tensor] = None
    acc: Optional[torch.Tensor] = None
    phys_props: PhysicalProps | None = None


@dataclass
class ParticleSystem2DTensor(System2DTensor):
    sim_props: SimulationProps | None = None
    step: Optional[int] = 0

    def __post_init__(self):
        self.vel = self.vel if self.vel is not None else torch.zeros_like(self.pos)
        self.acc = self.acc if self.acc is not None else torch.zeros_like(self.pos)
        # self.enabled = torch.ones(n_particles, dtype=torch.bool, device=device)

    @classmethod
    def initialize_particles_in_circle(cls, n_particles, R, device, random_seed=None):
        generator = None
        if random_seed:
            generator = torch.Generator(device=device)
            generator.manual_seed(random_seed)

        theta = 2 * math.pi * torch.rand(n_particles, device=device, generator=generator)

        r = R * torch.sqrt(torch.rand(n_particles, device=device, generator=generator))

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        pos = torch.stack((x, y), dim=1)  # shape: [n_particles, 2]
        return pos

    @classmethod
    def solid_circle_confinment(cls, positions, velocities, radio: float):
        pos = positions.clone()
        vel = velocities.clone()

        r_mag = torch.linalg.norm(pos, dim=1)
        collided = r_mag > radio

        if collided.any():
            n = pos[collided] / r_mag[collided].unsqueeze(1)
            v_collided = vel[collided]
            dot_products = torch.sum(v_collided * n, dim=1, keepdim=True)
            vel[collided] = v_collided - 2 * dot_products * n
            pos[collided] = n * radio

        return pos, vel

    def _step_plus_one(self):
        self.step += 1
        return self.step

    def update(self, pos, vel, acc):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self._step_plus_one()

    @property
    def to_dict(self):
        return [{
            "step": self.step,
            "p_idx": idx,
            "rx": pos[0].item(),
            "ry": pos[1].item(),
            "charge": self.phys_props.q
        } for idx, pos in enumerate(self.pos)]

    # def mean_velocity(self):
    #     return torch.norm(self.vel, dim=1).mean()
    #
    # def mean_acceleration(self):
    #     mask = torch.norm(self.pos, dim=1) < self.r_confinement if self.r_confinement > 0 else torch.ones(self.n_particles,
    #                                                                                             device=device,
    #                                                                                             dtype=torch.bool)
    #     return torch.norm(self.acc[mask], dim=1).mean()


if __name__ == "__main__":
    RADIO = 1.0
    N_PARTICLES = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ps = ParticleSystem2DTensor(
        pos=ParticleSystem2DTensor.initialize_particles_in_circle(
            n_particles=N_PARTICLES,
            R=RADIO,
            device=device,
        ),
        sim_props=SimulationProps(
            r_confinement=RADIO,
            beta=0.0,
            dt=1.0,
            g=9.8,
            k=1.0,
            min_vel=0.0,
            k_confinement=0.0,
        ),
        phys_props=PhysicalProps(
            q=torch.full((N_PARTICLES, 1), 2.0, device=device),
            m=torch.full((N_PARTICLES, 1), 3.0, device=device),
        )
    )
    print(ps.pos)
    print(ps.to_dict)
