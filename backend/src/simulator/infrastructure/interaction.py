from dataclasses import dataclass

import torch

from backend.src.simulator.domain.interfaces import Interaction, InteractionDecorator
from backend.src.simulator.infrastructure.queries import GenericInteractionQuery, GenericInteractionResponse

SECURE_DIVISION_CONSTANT = 1e-9


class PairElectrostaticInteraction(
    Interaction[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        r = query.positions.unsqueeze(1) - query.positions.unsqueeze(0)  # [n, n, 2]
        dist = torch.norm(r, dim=2, keepdim=True) + SECURE_DIVISION_CONSTANT  # 1e-9 para que no haya division entre 0
        # print("DIST ", dist)
        ff = (1.0 / dist) ** 3
        aceleration = (r * ff).sum(dim=1)
        aceleration *= query.sim_props.k * query.phys_props.q ** 2 # TODO multiplicar por la carga de cada particula
        aceleration /= query.phys_props.m

        return GenericInteractionResponse(
            acceleration=aceleration,
        )


class BarrasInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, positions):
        pass


@dataclass
class PotencialWallInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        """
        acel: Tensor([ax, ay])
        pos: Tensor([x, y])
        returns: Tensor([new_ax, new_ay])
        """
        parabolic_acceleration = query.sim_props.k_confinement * query.positions
        parabolic_acceleration /= query.phys_props.m
        return GenericInteractionResponse(
            acceleration=interaction_response.acceleration - parabolic_acceleration
        )


class FrictionInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        acceleration = interaction_response.acceleration
        acceleration -= query.sim_props.beta * query.velocity
        return GenericInteractionResponse(
            acceleration=acceleration,
        )
