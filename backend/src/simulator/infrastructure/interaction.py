"""Interacciones del simulador (Coulomb, WCA, paredes, friccion, gravedad).

Cota de confinamiento con defaults (k=10, q=1, R=12): ~1900 particulas con
margen >=10x sobre el soft-cap; ~2700 si se tolera el knee del WCA. Ver
docs/volumen-excluido-wca.md, seccion 7.1.
"""

from dataclasses import dataclass

import torch

from src.common.domain.enums import ConfinementType
from src.simulator.domain.interfaces import Interaction, InteractionDecorator
from src.simulator.infrastructure.helpers.interaction import build_base_response, accumulate, apply_overrides
from src.simulator.infrastructure.queries import GenericInteractionQuery, GenericInteractionResponse

SECURE_DIVISION_CONSTANT = 1e-9

# Defaults de operación de las interacciones WCA: heurísticos de estabilidad
# (no constantes físicas), overridables por parámetros del decorator. Detalles
# y límites en docs/volumen-excluido-wca.md sección 7.1.
WALL_EPSILON_KCONF_RATIO = 150.0  # rigidez de pared ~150x la trampa armónica
WALL_SIGMA_R_FRACTION = 0.15  # sigma de pared como fracción de R
WCA_WALL_FORCE_MAX_DEFAULT = 20000.0  # umbral del soft-cap de la pared
PAIR_WCA_SIGMA_R_FRACTION = 0.05  # sigma par-par como fracción de R
PAIR_WCA_FORCE_MAX_DEFAULT = 5000.0  # umbral del soft-cap par-par


def _soft_cap(force: torch.Tensor, force_max: float) -> torch.Tensor:
    """Saturación suave (C-infinito) para el WCA: sin meseta ni kink.

    La repulsión 1/d**13 diverge al contacto y no es integrable con dt finitos.
    Cap duro = meseta plana (derivada 0) que ensucia la solución; este soft-cap
    coincide con la fuerza pura para F << force_max y tiende asintóticamente a
    force_max, de modo que solo se desvía en el régimen divergente.

    ADVERTENCIA: es una REGULARIZACIÓN NUMÉRICA del integrador, no un modelo
    físico. Solo acota la rigidez local omega = sqrt(F'/m) para cumplir la cota
    de estabilidad omega*dt <= 2 de Velocity-Verlet. Limitaciones:
    - El potencial efectivo V(r) = -int F_sat dr queda ACOTADO (barrera finita).
      A diferencia del WCA puro (V -> inf, impenetrable), un par con energía
      cinética relativa mayor que la barrera podría en principio atravesarse:
      el "volumen excluido" deja de ser excluyente duro.
    - Solo se comporta como WCA si F << force_max (desviación x/(1+x)). Con
      F ~ force_max modifica la física en silencio (estructura, presión) y las
      observables ya no son las del potencial WCA.
    - Aplica SOLO al término WCA: el Coulomb (PairElectrostaticInteraction) no
      está capado, así que la fuerza total de un par muy solapado puede seguir
      creciendo por encima de force_max vía 1/d**2.

    Métodos alternativos más robustos / mejor justificados físicamente:
    - Bajar el dt: la respuesta correcta por defecto (el overshoot de Verlet
      escala con omega*dt) y la única que no cambia el potencial.
    - Potencial shifted-force / force-switch al corte (Toxvaerd & Dyre 2011;
      GROMACS force-switch; LAMMPS pair_style lj/smooth): suaviza la fuerza
      en el corte sin tocar el régimen de contacto.
    - Resolver solapamientos iniciales con minimización (steepest-descent
      hasta criterio de fuerza máxima; LAMMPS fix nve/limit) o con
      pair_style soft + fix adapt y luego activar el WCA real.
    - Núcleo blando que no permite cruce (soft-core LJ, Beutler et al. 1994)
      o potenciales blandos de DPD (Groot & Warren 1997).
    - Softening completo de la divergencia tipo Plummer en N-body (Aarseth
      1963): misma idea de acotar 1/r^2, forma distinta y conservativa.

    Uso seguro: force_max varios órdenes por encima de las fuerzas típicas y
    verificación de que las observables no cambian al variar force_max (ver
    docs/volumen-excluido-wca.md, sección 7.1).
    """
    x = force / force_max
    return force_max * x / (1.0 + x)


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


@dataclass
class PairWCAInteractionDecorator(
    InteractionDecorator[
        GenericInteractionQuery,
        GenericInteractionResponse,
    ]
):
    # Exclusión de volumen par-par (Weeks-Chandler-Andersen): repulsión de corto
    # alcance que impone una distancia mínima entre partículas y elimina la
    # singularidad 1/d**2 del Coulomb (causa del Infinity con muchas partículas).
    #   V(d) = 4*eps*[(sigma/d)^12 - (sigma/d)^6] + eps    d < d_c = 2^(1/6)*sigma
    #   F(d) = (24*eps/sigma)*[2*(sigma/d)^13 - (sigma/d)^7]
    epsilon: float | None = None
    sigma: float | None = None
    d_cutoff: float | None = None
    force_max: float | None = None

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        positions = query.positions
        n_particles = positions.shape[0]

        r = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 2]
        dist = torch.norm(r, dim=2, keepdim=True) + SECURE_DIVISION_CONSTANT  # [N, N, 1]

        R = query.sim_props.r_confinement
        epsilon = self.epsilon if self.epsilon is not None else query.sim_props.k_confinement
        sigma = self.sigma if self.sigma is not None else PAIR_WCA_SIGMA_R_FRACTION * R
        d_cut = self.d_cutoff if self.d_cutoff is not None else (2.0 ** (1.0 / 6.0)) * sigma
        # Soft-cap: el 1/d**13 diverge al contacto; se satura suavemente (sin
        # meseta ni kink) para que el paso de integración siga siendo estable.
        # Margen del umbral: ver docs/volumen-excluido-wca.md sección 7.1.
        force_max = self.force_max if self.force_max is not None else PAIR_WCA_FORCE_MAX_DEFAULT

        # WCA puro en float64 (evita overflow de s**13) y soft-cap.
        s = sigma / dist.double()
        force_mag = (24.0 * epsilon / sigma) * (2.0 * s ** 13 - s ** 7)
        force_mag = _soft_cap(force_mag, force_max).to(dist.dtype)
        force_mag = torch.where(
            dist < d_cut,
            force_mag,
            torch.zeros_like(dist),
        )

        # Eliminar la auto-interacción (diagonal).
        self_mask = torch.eye(
            n_particles,
            dtype=torch.bool,
            device=positions.device,
        ).unsqueeze(-1)
        force_mag = force_mag.masked_fill(self_mask, 0.0)

        direction = r / torch.clamp(dist, min=SECURE_DIVISION_CONSTANT)
        force = direction * force_mag  # [N, N, 2]
        acceleration = force.sum(dim=1) / query.phys_props.m
        return accumulate(
            response=interaction_response,
            component="pair_wca",
            contribution=acceleration,
            mass=query.phys_props.m,
        )


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
    # distancia a la pared d = |R - r|, puramente repulsivo.
    #   V(d) = 4*eps*[(sigma/d)^12 - (sigma/d)^6] + eps    d < d_c = 2^(1/6)*sigma
    #   F(d) = (24*eps/sigma)*[2*(sigma/d)^13 - (sigma/d)^7]
    # Además rescata cualquier partícula con r > R con fuerza lineal en el
    # overrun (estable, omega = sqrt(2*eps)) para que nunca queden fuera.
    epsilon: float | None = None
    sigma: float | None = None
    d_cutoff: float | None = None
    force_max: float | None = None

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        positions = query.positions
        r = torch.norm(positions, dim=1, keepdim=True)
        R = query.sim_props.r_confinement
        # Misma rigidez que la pared armónica, para retener el arranque de
        # expansión de Coulomb y no dejar escapar la frontera.
        epsilon = self.epsilon if self.epsilon is not None else WALL_EPSILON_KCONF_RATIO * query.sim_props.k_confinement
        sigma = self.sigma if self.sigma is not None else WALL_SIGMA_R_FRACTION * R
        d_cut = self.d_cutoff if self.d_cutoff is not None else (2.0 ** (1.0 / 6.0)) * sigma
        # Soft-cap de la cáscara: satura suavemente el 1/d**13 sin meseta ni
        # kink; el régimen normal queda idéntico al WCA puro (F << force_max).
        force_max = self.force_max if self.force_max is not None else WCA_WALL_FORCE_MAX_DEFAULT

        # El WCA solo actúa dentro de la cáscara (R - d_c < r < R + d_c); fuera
        # de ella la fuerza es 0, y una partícula que la cruzó derivaba libre.
        # El overrun rescata cualquier partícula con r > R con fuerza lineal en
        # el overrun (estable, omega = sqrt(2*eps) como la pared armónica).
        overrun = torch.clamp(r - R, min=0.0)
        distance = torch.abs(R - r)
        inside_shell = (distance < d_cut) & (overrun <= 0.0)
        # WCA puro en float64 (evita overflow de s**13) y soft-cap.
        s = sigma / torch.clamp(distance, min=SECURE_DIVISION_CONSTANT).double()
        shell_force = (24.0 * epsilon / sigma) * (2.0 * s ** 13 - s ** 7)
        shell_force = _soft_cap(shell_force, force_max).to(r.dtype)
        shell_force = torch.where(
            inside_shell,
            shell_force,
            torch.zeros_like(r),
        )
        rescue_force = 2.0 * epsilon * overrun
        force_mag = torch.where(
            overrun > 0.0,
            rescue_force,
            shell_force,
        )
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
    #   |F(d)| = 2*eps*(d_c - d), lineal, radial hacia adentro
    # Además rescata cualquier partícula con r > R con fuerza lineal en el
    # overrun (estable, omega=sqrt(2*eps)) para que nunca queden fuera del disco.
    epsilon: float | None = None
    d_cutoff: float | None = None

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        interaction_response: GenericInteractionResponse = super().compute_aceleration(query)
        positions = query.positions
        r = torch.norm(positions, dim=1, keepdim=True)
        R = query.sim_props.r_confinement
        # Heurístico: la pared debe ser ~150x más rígida que la trampa armónica
        # (que vale k_confinement) para confinar en la cáscara corta; ajustable.
        epsilon = self.epsilon if self.epsilon is not None else WALL_EPSILON_KCONF_RATIO * query.sim_props.k_confinement
        sigma = WALL_SIGMA_R_FRACTION * R
        d_cut = self.d_cutoff if self.d_cutoff is not None else (2.0 ** (1.0 / 6.0)) * sigma

        overrun = torch.clamp(r - R, min=0.0)
        distance = R - r
        inside_shell = (distance >= 0.0) & (distance < d_cut)
        shell_force = 2.0 * epsilon * (d_cut - distance)
        rescue_force = 2.0 * epsilon * overrun
        force_mag = torch.where(
            overrun > 0.0,
            rescue_force,
            shell_force,
        )
        force_mag = torch.where(inside_shell | (overrun > 0.0), force_mag, torch.zeros_like(force_mag))
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
    interactions = PairWCAInteractionDecorator(interactions)
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
