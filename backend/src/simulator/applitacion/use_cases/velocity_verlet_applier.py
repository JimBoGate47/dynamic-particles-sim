from dataclasses import dataclass
from typing import TypeVar, Generic

from src.common.domain.entities.properties import SimulationProps
from src.common.domain.interfaces import UseCase
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.domain.interfaces import Interaction, SystemRestriction
from src.simulator.infrastructure.queries import (
    GenericInteractionResponse,
    GenericInteractionQuery,
    PositionRestrictionQuery,
    PositionRestrictionResponse,
)

T = TypeVar("T")


@dataclass
class VelocityVerletApplier(UseCase, Generic[T]):
    particle_system: ParticleSystem2DTensor
    sim_props: SimulationProps
    interactions: Interaction
    restriction: SystemRestriction | None = None

    async def execute(self, *args, **kwargs) -> GenericInteractionResponse:
        half_vel = self.calculate_half_vel(
            vel=self.particle_system.vel,
            acc=self.particle_system.acc,
            dt=self.sim_props.dt,
        )

        new_pos = self.calculate_new_pos(
            pos=self.particle_system.pos,
            half_vel=half_vel,
            dt=self.sim_props.dt,
        )
        new_pos = self._in_place_position_restriction(
            old_positions=self.particle_system.pos,
            new_positions=new_pos,
        )

        response: GenericInteractionResponse = self.interactions.compute_aceleration(
            query=GenericInteractionQuery(
                positions=new_pos,
                velocity=half_vel,
                sim_props=self.sim_props,
                phys_props=self.particle_system.phys_props,
            ),
        )
        new_acc = response.acceleration

        new_vel = self.calculate_new_vel(
            half_vel=half_vel,
            new_acc=new_acc,
            dt=self.sim_props.dt,
        )
        # Los decoradores de pared (reflexión dura) pueden sobreescribir la
        # posición/velocidad integradas para corregir el cruce de la pared.
        if response.velocity is not None:
            new_vel = response.velocity
        if response.positions is not None:
            new_pos = response.positions

        self.particle_system.update(
            pos=new_pos,
            vel=new_vel,
            acc=new_acc,
        )
        return response

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
