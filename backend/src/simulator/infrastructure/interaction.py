import warnings
from dataclasses import dataclass

import torch

from src.simulator.domain.interfaces import Interaction, InteractionDecorator
from src.simulator.infrastructure.queries import GenericInteractionQuery, GenericInteractionResponse

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
        raise NotImplementedError()


class GravityInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        gravity_force = query.phys_props.m * query.sim_props.g
        gravity_acceleration = torch.zeros_like(interaction_response.acceleration)
        gravity_acceleration[:, 1] = - gravity_force.squeeze() / query.phys_props.m.squeeze()
        acceleration = interaction_response.acceleration + gravity_acceleration
        return GenericInteractionResponse(
            positions=query.positions,
            velocity=query.velocity,
            acceleration=acceleration,
        )


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
        potential_force = - query.sim_props.k_confinement * query.positions
        potential_acceleration = potential_force / query.phys_props.m
        return GenericInteractionResponse(
            positions=interaction_response.positions,
            velocity=interaction_response.velocity,
            acceleration=interaction_response.acceleration + potential_acceleration
        )


@dataclass
class Potencial4WallInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        # The potential is U = (1/4) * k * |r|^4
        # The force is F = -grad(U) = -k * |r|^2 * r_vec
        # The acceleration is a = F/m = - (k/m) * |r|^2 * r_vec
        positions = query.positions
        r_squared = torch.sum(positions ** 2, dim=1, keepdim=True)
        potential_force = - query.sim_props.k_confinement * r_squared * positions
        potential_acceleration = potential_force / query.phys_props.m
        return GenericInteractionResponse(
            positions=interaction_response.positions,
            velocity=interaction_response.velocity,
            acceleration=interaction_response.acceleration + potential_acceleration
        )


@dataclass
class Potencial8WallInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        # The potential is U = (1/8) * k * |r|^8
        # The force is F = -grad(U) = -k * |r|^6 * r_vec
        # The acceleration is a = F/m = - (k/m) * |r|^2 * r_vec
        positions = query.positions
        r_squared = torch.sum(positions ** 2, dim=1, keepdim=True)
        # r^6 = (r^2)^3
        r_sixth = r_squared ** 3
        potential_force = - query.sim_props.k_confinement * r_sixth * positions
        potential_acceleration = potential_force / query.phys_props.m
        return GenericInteractionResponse(
            positions=interaction_response.positions,
            velocity=interaction_response.velocity,
            acceleration=interaction_response.acceleration + potential_acceleration
        )


class HardWallInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        """
        Reflects particles upon collision with a wall of radius R.
        For highly charged particles, very small time steps (dt) will be required.
        Very large velocities will make the system unstable.
        """
        warnings.warn("""
        This class may be unnecessary; consider using a hard wall by applying a potential V = k * r^n with n → infinity.
        """)
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        pos = query.positions.clone()
        vel = query.velocity.clone()

        r_mag = torch.linalg.norm(pos, dim=1)
        collided = r_mag > query.sim_props.r_confinement

        if collided.any():
            n = pos[collided] / r_mag[collided].unsqueeze(1)
            v_collided = vel[collided]
            dot_products = torch.sum(v_collided * n, dim=1, keepdim=True)
            vel[collided] = v_collided - 2 * dot_products * n
            pos[collided] = n * query.sim_props.r_confinement

        return GenericInteractionResponse(
            positions=pos,
            velocity=vel,
            acceleration=interaction_response.acceleration,
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
