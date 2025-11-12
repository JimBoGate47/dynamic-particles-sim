import math
from dataclasses import dataclass
from typing import Optional

import torch

from backend.src.common.domain.entities.properties import SimulationProps, PhysicalProps
from backend.src.common.domain.types.properties import PhysicalProperties, SimulationProperties
from backend.src.simulator.infrastructure.interaction import (
    PairElectrostaticInteraction,
    PotencialWallInteractionDecorator,
    FrictionInteractionDecorator,
)
from backend.src.simulator.infrastructure.queries import GenericInteractionQuery, GenericInteractionResponse

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
class ParticleSystem2DTensor:
    pos: torch.Tensor
    phys_props: PhysicalProps
    sim_props: SimulationProps
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
        new_pos = self.calculate_new_pos()
        half_vel = self.calculate_half_vel()

        response: GenericInteractionResponse = interactions.compute_aceleration(
            query=GenericInteractionQuery(
                positions=new_pos,
                velocity=half_vel,  # NOTE Opcion A: self.vel Opcion B: half_velocity (more accurate)
                sim_props=self.sim_props,
                phys_props=self.phys_props,
            ),
        )
        new_acc = response.acceleration

        # TODO mover a un class decorador?, ver PotencialWallInteractionDecorator
        # new_pos, vel_half = self.solid_circle_confinment(
        #     positions=new_pos,
        #     velocities=vel_half,
        #     radio=self.sim_props.r_confinement,
        # )

        new_vel = self.calculate_new_vel(half_vel=half_vel, new_acc=new_acc)

        self.pos = new_pos
        self.vel = new_vel
        self.acc = new_acc

        self.step += 1

    def calculate_new_pos(self) -> torch.Tensor:
        return self.pos + self.vel * self.sim_props.dt + 0.5 * self.acc * (self.sim_props.dt ** 2)

    def calculate_half_vel(self) -> torch.Tensor:
        return self.vel + 0.5 * self.acc * self.sim_props.dt

    def calculate_new_vel(self, half_vel, new_acc):
        return half_vel + 0.5 * new_acc * self.sim_props.dt

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
    #     mask = torch.norm(self.pos, dim=1) < self.r_confinement if self.r_confinement > 0 else torch.ones(self.n_particles,
    #                                                                                             device=device,
    #                                                                                             dtype=torch.bool)
    #     return torch.norm(self.acc[mask], dim=1).mean()


if __name__ == "__main__":
    RADIO = 1.0
    ps = ParticleSystem2DTensor(
        # pos=torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device),  # torch.randn(3, 2, device=device),
        # pos=torch.randn(3, 2, device=device),
        pos=ParticleSystem2DTensor.initialize_particles_in_circle(
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
