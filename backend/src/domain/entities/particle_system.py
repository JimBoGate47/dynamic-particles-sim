import math
from dataclasses import dataclass
from typing import Optional

import torch

from backend.src.domain.entities.properties import PhysicalProperties, SimulationProperties
from backend.src.infrastructure.interaction import (
    PairElectrostaticInteraction,
    PotencialWallInteractionDecorator,
    FrictionInteractionDecorator,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def compute_acceleration(pos):
#     r = pos.unsqueeze(1) - pos.unsqueeze(0)
#     dist = torch.norm(r, dim=2, keepdim=True) + 1e-9
#     ff = (1.0 / dist) ** 3
#     acc = (r * ff).sum(dim=1)
#     return acc

# def wall(vel, pos): # TODO revisame si es necesario
#     raise DeprecationWarning
#     """
#     vel: np.array([vx, vy])
#     pos: np.array([x, y])
#     returns: np.array([vf_x, vf_y])
#     """
#     rinv = 1.0 / np.sqrt(pos[0] ** 2 + pos[1] ** 2)
#     dot = vel[0] * pos[1] - vel[1] * pos[0]
#     vf_x = -vel[0] + 2 * dot * pos[1] * rinv * rinv
#     vf_y = -vel[1] - 2 * dot * pos[0] * rinv * rinv
#     return np.array([vf_x, vf_y])


interactions = PairElectrostaticInteraction()
interactions = PotencialWallInteractionDecorator(interactions)
interactions_plus_friction = FrictionInteractionDecorator(interactions)


@dataclass
class ParticleSystem2D:
    pos: torch.Tensor
    phys_props: PhysicalProperties
    sim_props: SimulationProperties
    vel: Optional[torch.Tensor] = None
    acc: Optional[torch.Tensor] = None
    step: Optional[int] = 0

    def __post_init__(self):
        self.vel = self.vel or torch.zeros_like(self.pos)
        self.acc = self.acc or torch.zeros_like(self.pos)
        # self.enabled = torch.ones(n_particles, dtype=torch.bool, device=device)

    @classmethod
    def initialize_particles_in_circle(cls, n_particles, R, device):
        theta = 2 * math.pi * torch.rand(n_particles, device=device)

        r = R * torch.sqrt(torch.rand(n_particles, device=device))

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        pos = torch.stack((x, y), dim=1)  # shape: [n_particles, 2]
        return pos

    def velocity_verlet_step(self):
        acc = interactions.compute_aceleration(
            positions=self.pos,
            phys_props=self.phys_props,
        )
        # print("POS0: ", self.pos)
        # print("VEL0: ", self.vel)
        # print("ACC0: ", self.acc)

        vel_half = self.vel + 0.5 * acc * self.sim_props.dt
        new_pos = self.pos + vel_half * self.sim_props.dt

        # TODO mover a un class decorador?, ver PotencialWallInteractionDecorator
        # new_pos, vel_half = self.solid_circle_confinment(
        #     positions=new_pos,
        #     velocities=vel_half,
        #     radio=self.sim_props.r_confin,
        # )

        new_acc = interactions_plus_friction.compute_aceleration(
            positions=new_pos,
            beta=self.sim_props.beta,
            velocity=vel_half,
            phys_props=self.phys_props,
        )

        new_vel = vel_half + 0.5 * new_acc * self.sim_props.dt

        self.pos = new_pos
        self.vel = new_vel
        self.acc = new_acc

        self.step += 1

        # print("POS1: ", self.pos)
        # print("VEL1: ", self.vel)
        # print("ACC1: ", self.acc)

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
    #     mask = torch.norm(self.pos, dim=1) < self.r_confin if self.r_confin > 0 else torch.ones(self.n_particles,
    #                                                                                             device=device,
    #                                                                                             dtype=torch.bool)
    #     return torch.norm(self.acc[mask], dim=1).mean()


if __name__ == "__main__":
    RADIO = 1.0
    ps = ParticleSystem2D(
        # pos=torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device),  # torch.randn(3, 2, device=device),
        # pos=torch.randn(3, 2, device=device),
        pos=ParticleSystem2D.initialize_particles_in_circle(
            n_particles=3,
            R=RADIO,
            device=device,
        ),
        sim_props=SimulationProperties(
            r_confin=RADIO,
            beta=0.0,
            dt=1.0,
        ),
        phys_props=PhysicalProperties(
            q=2.0,
            m=3.0,
        )
    )
    print(ps.pos)
    ps.velocity_verlet_step()
    print(ps.pos)
    print(ps.to_dict)
