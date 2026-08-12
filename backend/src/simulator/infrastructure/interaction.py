from dataclasses import dataclass

import torch

from src.common.domain.enums import ConfinementType
from src.simulator.domain.interfaces import Interaction, InteractionDecorator
from src.simulator.infrastructure.helpers.interaction import build_base_response, accumulate, apply_overrides
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

        return build_base_response(
            component="electrostatic",
            acceleration=aceleration,
            mass=query.phys_props.m,
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
        return accumulate(
            response=interaction_response,
            component="gravity",
            contribution=gravity_acceleration,
            mass=query.phys_props.m,
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
        return accumulate(
            response=interaction_response,
            component="potencial_wall",
            contribution=potential_acceleration,
            mass=query.phys_props.m,
        )


@dataclass
class WCAWallInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    # Sección 4.1: potencial WCA (Weeks–Chandler–Andersen) dependiente de la
    # distancia a la pared d = |R - r|, puramente repulsivo, con fuerza acotada.
    #   V(d) = 4*eps*[(sigma/d)^12 - (sigma/d)^6] + eps    d < d_c = 2^(1/6)*sigma
    #   F(d) = (24*eps/sigma)*[2*(sigma/d)^13 - (sigma/d)^7]
    epsilon: float | None = None
    sigma: float | None = None
    d_cutoff: float | None = None
    force_max: float | None = None

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        positions = query.positions
        r = torch.norm(positions, dim=1, keepdim=True)
        R = query.sim_props.r_confinement
        epsilon = self.epsilon if self.epsilon is not None else query.sim_props.k_confinement
        sigma = self.sigma if self.sigma is not None else 0.1 * R
        d_cut = self.d_cutoff if self.d_cutoff is not None else (2.0 ** (1.0 / 6.0)) * sigma
        # Tope de fuerza por defecto: suficiente para frenar a las partículas con
        # dt típico (dt=0.1) sin "fling"; valores >~200 inyectan demasiado momento.
        force_max = self.force_max if self.force_max is not None else 100.0

        distance = torch.abs(R - r)
        active = distance < d_cut
        # s_cap evita overflow de s**13 en float32 (ver sección 5.2 del doc)
        s_cap = (
                        force_max
                        * max(sigma, SECURE_DIVISION_CONSTANT)
                        / (48.0 * max(epsilon, SECURE_DIVISION_CONSTANT))
                ) ** (1.0 / 13.0)
        s = torch.clamp(sigma / torch.clamp(distance, min=SECURE_DIVISION_CONSTANT), max=s_cap)
        force_mag = (24.0 * epsilon / sigma) * (2.0 * s ** 13 - s ** 7)
        force_mag = torch.clamp(force_mag, max=force_max)
        force_mag = torch.where(active, force_mag, torch.zeros_like(force_mag))
        inward = -positions / torch.clamp(r, min=SECURE_DIVISION_CONSTANT)
        wall_acceleration = (force_mag * inward) / query.phys_props.m
        return accumulate(
            response=interaction_response,
            component="wca_wall",
            contribution=wall_acceleration,
            mass=query.phys_props.m,
        )


@dataclass
class HarmonicWallInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    # Sección 4.3: pared armónica (resorte repulsivo) dependiente de d = |R - r|.
    #   V(d) = eps*(d - d_c)^2    d < d_c
    #   |F(d)| = 2*eps*(d_c - d)  lineal, acotada, radial hacia adentro
    epsilon: float | None = None
    d_cutoff: float | None = None

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        positions = query.positions
        r = torch.norm(positions, dim=1, keepdim=True)
        R = query.sim_props.r_confinement
        # Heurístico: la pared debe ser ~100x más rígida que la trampa armónica
        # (que vale k_confinement) para confinar en la cáscara corta; ajustable.
        epsilon = self.epsilon if self.epsilon is not None else 100.0 * query.sim_props.k_confinement
        sigma = 0.1 * R
        d_cut = self.d_cutoff if self.d_cutoff is not None else (2.0 ** (1.0 / 6.0)) * sigma

        distance = torch.abs(R - r)
        active = distance < d_cut
        force_mag = 2.0 * epsilon * (d_cut - distance)
        force_mag = torch.clamp(force_mag, min=0.0)
        force_mag = torch.where(active, force_mag, torch.zeros_like(force_mag))
        inward = -positions / torch.clamp(r, min=SECURE_DIVISION_CONSTANT)
        wall_acceleration = (force_mag * inward) / query.phys_props.m
        return accumulate(
            response=interaction_response,
            component="harmonic_wall",
            contribution=wall_acceleration,
            mass=query.phys_props.m,
        )


@dataclass
class HardWallEventDrivenInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    # Sección 6.3: pared dura exacta por reflexión especular event-driven.
    # Resuelve t_c en |r0 + v*t_c| = R (ecuación cuadrática, sección 3.1),
    # refleja en el punto de contacto y continúa el tiempo restante.
    # Corrige el tunneling de HardWallInteractionDecorator (que solo revisaba
    # la posición final) y rescata partículas que quedaron fuera de la pared.
    padding: float = 0.0

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        pos = query.positions.clone()
        vel = query.velocity.clone()
        R = query.sim_props.r_confinement + self.padding
        dt = query.sim_props.dt

        # Coeficientes de la cuadrática a*t^2 + 2*b*t + c = 0
        a = torch.sum(vel ** 2, dim=1)
        b = torch.sum(pos * vel, dim=1)
        c = torch.sum(pos ** 2, dim=1) - R ** 2
        disc = b ** 2 - a * c

        eps = torch.finfo(a.dtype).eps
        a_safe = torch.where(a > eps, a, torch.ones_like(a))
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t1 = (-b - sqrt_disc) / a_safe
        t2 = (-b + sqrt_disc) / a_safe
        inf = torch.full_like(t1, float("inf"))
        t_coll = torch.minimum(
            torch.where(t1 >= 0.0, t1, inf),
            torch.where(t2 >= 0.0, t2, inf),
        )

        # Punto de contacto y su normal para las partículas dentro del disco.
        tc = torch.where(torch.isfinite(t_coll), t_coll, torch.zeros_like(t_coll))
        r_c = pos + vel * tc.unsqueeze(1)
        n = r_c / torch.clamp(torch.linalg.norm(r_c, dim=1, keepdim=True), min=eps)
        dot = torch.sum(vel * n, dim=1, keepdim=True)

        # Solo se refleja si la partícula intenta salir (velocidad radial saliente).
        leaving = dot.squeeze(1) > 0.0
        has_collision = (
                (a > eps)
                & (disc >= 0.0)
                & torch.isfinite(t_coll)
                & (t_coll <= dt)
                & leaving
        )

        # Partículas que colisionan: avanzar a la pared, reflejar, continuar.
        tc = torch.where(has_collision, t_coll, torch.zeros_like(t_coll))
        r_c = pos + vel * tc.unsqueeze(1)
        n = r_c / torch.clamp(torch.linalg.norm(r_c, dim=1, keepdim=True), min=eps)
        dot = torch.sum(vel * n, dim=1, keepdim=True)
        v_reflected = vel - 2.0 * dot * n
        remaining = dt - tc
        pos_collided = r_c + v_reflected * remaining.unsqueeze(1)

        # Partículas sin colisión: propagación libre.
        pos_moved = pos + vel * dt

        # Rescate de partículas fuera de la pared: snap a la pared y, si se
        # alejan, invertir solo la componente radial saliente.
        r0 = torch.linalg.norm(pos, dim=1)
        outside = r0 > R
        rescue = outside & ~has_collision
        n_out = pos / torch.clamp(r0.unsqueeze(1), min=eps)
        dot_out = torch.sum(vel * n_out, dim=1, keepdim=True)
        v_rescued = vel - 2.0 * torch.clamp(dot_out, min=0.0) * n_out
        pos_rescued = n_out * R

        new_pos = torch.where(has_collision.unsqueeze(1), pos_collided, pos_moved)
        new_vel = torch.where(has_collision.unsqueeze(1), v_reflected, vel)
        new_pos = torch.where(rescue.unsqueeze(1), pos_rescued, new_pos)
        new_vel = torch.where(rescue.unsqueeze(1), v_rescued, new_vel)

        return apply_overrides(
            response=interaction_response,
            positions=new_pos,
            velocity=new_vel,
        )


class FrictionInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        force_friction = - query.sim_props.beta * query.velocity
        return accumulate(
            response=interaction_response,
            component="friction",
            contribution=force_friction / query.phys_props.m,
            mass=query.phys_props.m,
        )


def build_interactions(
        add_gravity: bool = False,
        wall: ConfinementType = ConfinementType.HARMONIC,
) -> Interaction:
    interactions = PairElectrostaticInteraction()
    if wall == ConfinementType.POTENCIAL:
        interactions = PotencialWallInteractionDecorator(interactions)
    elif wall == ConfinementType.WCA:
        interactions = WCAWallInteractionDecorator(interactions)
    elif wall == ConfinementType.HARD_WALL:
        interactions = HardWallEventDrivenInteractionDecorator(interactions)
    else:
        interactions = HarmonicWallInteractionDecorator(interactions)
    interactions = FrictionInteractionDecorator(interactions)

    if add_gravity:
        interactions = GravityInteractionDecorator(interactions)
    return interactions
