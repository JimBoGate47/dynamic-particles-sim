from dataclasses import dataclass
from typing import TypeVar, Generic

from backend.src.common.domain.interfaces import UseCase
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from backend.src.simulator.domain.interfaces import Interaction, SystemRestriction
from backend.src.simulator.infrastructure.queries import (
    GenericInteractionResponse,
    GenericInteractionQuery,
    PositionRestrictionQuery,
    PositionRestrictionResponse,
)

T = TypeVar("T")


@dataclass
class VelocityVerletApplier(UseCase, Generic[T]):
    particle_system: ParticleSystem2DTensor
    interactions: Interaction
    restriction: SystemRestriction | None = None

    async def execute(self, *args, **kwargs):
        half_vel = self.calculate_half_vel(
            vel=self.particle_system.vel,
            acc=self.particle_system.acc,
            dt=self.particle_system.sim_props.dt,
        )

        new_pos = self.calculate_new_pos(
            pos=self.particle_system.pos,
            half_vel=half_vel,
            dt=self.particle_system.sim_props.dt,
        )
        new_pos = self._in_place_position_restriction(
            old_positions=self.particle_system.pos,
            new_positions=new_pos,
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

    def _in_place_position_restriction(
            self,
            old_positions,
            new_positions,
    ) -> T:
        if self.restriction:
            restriction_response: PositionRestrictionResponse = self.restriction.apply(
                query=PositionRestrictionQuery(
                    old_positions=old_positions,
                    new_positions=new_positions,
                )
            )
            return restriction_response.new_positions
        return new_positions

    @classmethod
    def calculate_new_pos(cls, pos, half_vel, dt) -> T:
        return pos + half_vel * dt

    @classmethod
    def calculate_half_vel(cls, vel, acc, dt) -> T:
        return vel + 0.5 * acc * dt

    @classmethod
    def calculate_new_vel(cls, half_vel, new_acc, dt) -> T:
        return half_vel + 0.5 * new_acc * dt
