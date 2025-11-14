from dataclasses import dataclass
from typing import TypeVar, Generic

from backend.src.common.domain.interfaces import UseCase
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from backend.src.simulator.domain.interfaces import Interaction
from backend.src.simulator.infrastructure.queries import GenericInteractionResponse, GenericInteractionQuery

T = TypeVar("T")


@dataclass
class VelocityVerletApplier(UseCase, Generic[T]):
    particle_system: ParticleSystem2DTensor
    interactions: Interaction

    async def execute(self, *args, **kwargs):
        new_pos = self.calculate_new_pos(
            pos=self.particle_system.pos,
            vel=self.particle_system.vel,
            acc=self.particle_system.acc,
            dt=self.particle_system.sim_props.dt,
        )
        half_vel = self.calculate_half_vel(
            vel=self.particle_system.vel,
            acc=self.particle_system.acc,
            dt=self.particle_system.sim_props.dt,
        )

        response: GenericInteractionResponse = self.interactions.compute_aceleration(
            query=GenericInteractionQuery(
                positions=new_pos,
                velocity=half_vel,  # NOTE Opcion A: self.vel Opcion B: half_velocity (more accurate)
                sim_props=self.particle_system.sim_props,
                phys_props=self.particle_system.phys_props,
            ),
        )
        new_acc = response.acceleration

        # TODO mover a un class decorador?, ver PotencialWallInteractionDecorator
        # new_pos, vel_half = self.solid_circle_confinment(
        #     positions=new_pos,
        #     velocities=vel_half,
        #     radio=self.sim_props.r_confinement,
        # )

        new_vel = self.calculate_new_vel(
            half_vel=half_vel,
            new_acc=new_acc,
            dt=self.particle_system.sim_props.dt,
        )

        self.particle_system.update(
            pos=new_pos,
            vel=new_vel,
            acc=new_acc,
        )

    @classmethod
    def calculate_new_pos(cls, pos, vel, acc, dt) -> T:
        return pos + cls.calculate_half_vel(vel, acc, dt) * dt

    @classmethod
    def calculate_half_vel(cls, vel, acc, dt) -> T:
        return vel + 0.5 * acc * dt

    @classmethod
    def calculate_new_vel(cls, half_vel, new_acc, dt) -> T:
        return half_vel + 0.5 * new_acc * dt
