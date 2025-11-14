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
        # r tiene shape [N, N, 2], donde N es el número de partículas.
        r = query.positions.unsqueeze(1) - query.positions.unsqueeze(0)

        # dist tiene shape [N, N, 1]
        dist = torch.norm(r, dim=2, keepdim=True) + SECURE_DIVISION_CONSTANT

        # --- INICIO DE LA MODIFICACIÓN ---

        # q tiene shape (N, 1). Lo convertimos a (1, N) para el producto.
        q_source = query.phys_props.q.transpose(0, 1)  # Shape: (1, N)

        # q_query tiene shape (N, 1)
        q_query = query.phys_props.q

        # El broadcasting (N, 1) * (1, N) resulta en una matriz de cargas (N, N)
        # Cada elemento (i, j) contiene q_i * q_j. La expandimos para que tenga la misma dimensión que 'dist'.
        charge_product = (q_query @ q_source).unsqueeze(-1)  # Shape: (N, N, 1)

        # ff sigue siendo la ley de la inversa del cuadrado
        ff = (1.0 / dist) ** 3

        # Multiplicamos la matriz de cargas por la fuerza base
        # y luego por el vector de dirección 'r'.
        force = r * ff * charge_product * query.sim_props.k

        # Sumamos todas las fuerzas que actúan sobre cada partícula (dim=1)
        # Dividimos por la masa individual de cada partícula.
        # m tiene shape (N, 1), por lo que la división es elemento a elemento.
        aceleration = force.sum(dim=1) / query.phys_props.m

        # --- FIN DE LA MODIFICACIÓN ---

        return GenericInteractionResponse(
            positions=query.positions,
            velocity=query.velocity,
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
            positions=interaction_response.positions,
            velocity=interaction_response.velocity,
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
        force_friction = - query.sim_props.beta * query.velocity
        acceleration += force_friction / query.phys_props.m
        return GenericInteractionResponse(
            positions=interaction_response.positions,
            velocity=interaction_response.velocity,
            acceleration=acceleration,
        )
