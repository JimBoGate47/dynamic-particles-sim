import asyncio

import pytest
import torch

from src.common.domain.entities.properties import PhysicalProps, SimulationProps
from src.common.domain.enums import ConfinementType
from src.simulator.applitacion.use_cases.velocity_verlet_applier import VelocityVerletApplier
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.domain.interfaces import Interaction
from src.simulator.infrastructure.interaction import (
    GravityInteractionDecorator,
    HardWallEventDrivenInteractionDecorator,
    HarmonicWallInteractionDecorator,
    PairWCAInteractionDecorator,
    WCAWallInteractionDecorator,
    build_interactions,
)
from src.simulator.infrastructure.queries import (
    GenericInteractionQuery,
    GenericInteractionResponse,
)

R_CONFINEMENT = 6.0
N_PARTICLES = 12
K_CONFINEMENT = 0.5
DT = 0.1


class NoopInteraction(Interaction):
    """Interacción de referencia que no aporta ninguna fuerza."""

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        return GenericInteractionResponse(acceleration=torch.zeros_like(query.positions))


class StubInteraction(Interaction):
    """Interacción con respuesta fija para aislar el comportamiento del decorador."""

    def __init__(self, acceleration=None, positions=None, velocity=None):
        self._acceleration = acceleration
        self._positions = positions
        self._velocity = velocity

    def compute_aceleration(self, query: GenericInteractionQuery) -> GenericInteractionResponse:
        return GenericInteractionResponse(
            positions=self._positions,
            velocity=self._velocity,
            acceleration=self._acceleration,
        )


def _sim_props(**overrides) -> SimulationProps:
    defaults = dict(
        g=0.0,
        k=10.0,
        min_vel=0.0,
        r_confinement=R_CONFINEMENT,
        k_confinement=K_CONFINEMENT,
        beta=0.6,
        dt=DT,
    )
    defaults.update(overrides)
    return SimulationProps(**defaults)


def _phys_props(n: int) -> PhysicalProps:
    return PhysicalProps(
        q=torch.full((n, 1), 1.0),
        m=torch.ones(n, 1),
    )


def _query(positions, velocity, sim_props=None) -> GenericInteractionQuery:
    n = positions.shape[0]
    return GenericInteractionQuery(
        positions=positions,
        velocity=velocity,
        sim_props=sim_props or _sim_props(),
        phys_props=_phys_props(n),
    )


def _ring_positions(radius: float, n: int = N_PARTICLES) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * torch.pi, n + 1)[:n]
    return radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def _radial_unit_velocity(positions: torch.Tensor, speed: float) -> torch.Tensor:
    return positions / torch.linalg.norm(positions, dim=1, keepdim=True) * speed


def _run(
    steps: int,
    positions,
    velocities,
    interactions,
    sim_props=None,
) -> ParticleSystem2DTensor:
    """Ejecuta `steps` pasos de Velocity-Verlet y devuelve el sistema final."""
    particle_system = ParticleSystem2DTensor(
        pos=positions.clone(),
        vel=velocities.clone(),
        acc=torch.zeros_like(positions),
        phys_props=_phys_props(positions.shape[0]),
        step=0,
    )
    applier = VelocityVerletApplier(
        particle_system=particle_system,
        sim_props=sim_props or _sim_props(),
        interactions=interactions,
    )
    for _ in range(steps):
        asyncio.run(applier.execute())
    return particle_system


def _max_radius(positions) -> float:
    return float(torch.linalg.norm(positions, dim=1).max())


class TestWCAWall:
    def test_no_wall_force_for_interior_particles(self):
        # Arrange
        positions = torch.tensor([[1.0, 0.0], [0.0, -2.0]])
        velocity = torch.zeros(2, 2)
        decorator = WCAWallInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.allclose(
            response.acceleration, torch.zeros_like(response.acceleration), atol=1e-6
        )

    def test_wall_force_points_inward_at_boundary(self):
        # Arrange
        positions = torch.tensor([[R_CONFINEMENT, 0.0]])
        velocity = torch.zeros(1, 2)
        decorator = WCAWallInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        acceleration = response.acceleration[0]
        assert torch.isfinite(acceleration).all()
        assert acceleration[0] < 0.0  # partícula en +x: fuerza hacia el centro
        assert torch.allclose(acceleration[1], torch.tensor(0.0), atol=1e-6)

    def test_confines_charged_particles_within_radius(self):
        # Arrange: pared WCA rígida (epsilon=150*k) sin cap de fuerza.
        # se requiere dt pequeño para que el escalón cumpla omega*dt < 2.
        positions = _ring_positions(radius=3.0)
        velocities = _radial_unit_velocity(positions, speed=0.3)
        interactions = build_interactions(wall="wca")
        sim_props = _sim_props(dt=0.005)

        # Act
        particle_system = _run(steps=300, positions=positions, velocities=velocities, interactions=interactions, sim_props=sim_props)

        # Assert
        assert torch.isfinite(particle_system.pos).all()
        assert _max_radius(particle_system.pos) <= R_CONFINEMENT + 1e-2

    def test_rescues_escaped_particle_inward(self):
        # Arrange: partícula fuera de la cáscara (r > R + d_cut) alejándose
        positions = torch.tensor([[7.0, 0.0]])
        velocity = torch.tensor([[1.0, 0.0]])
        decorator = WCAWallInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert: fuerza radial hacia adentro y finita
        assert torch.isfinite(response.acceleration).all()
        assert response.acceleration[0, 0] < 0.0
        assert torch.allclose(response.acceleration[0, 1], torch.tensor(0.0), atol=1e-6)

    def test_pulls_escaped_particle_back_during_integration(self):
        # Arrange: arranque fuera de la pared con velocidad saliente
        positions = torch.tensor([[10.0, 0.0]])
        velocities = torch.tensor([[1.0, 0.0]])
        interactions = build_interactions(wall="wca")
        sim_props = _sim_props(dt=0.005)

        # Act
        particle_system = _run(steps=300, positions=positions, velocities=velocities, interactions=interactions, sim_props=sim_props)

        # Assert: finita y de vuelta dentro del disco
        assert torch.isfinite(particle_system.pos).all()
        assert _max_radius(particle_system.pos) <= R_CONFINEMENT + 1e-2


class TestHarmonicWall:
    def test_wall_force_points_inward_at_boundary(self):
        # Arrange
        positions = torch.tensor([[R_CONFINEMENT, 0.0]])
        velocity = torch.zeros(1, 2)
        decorator = HarmonicWallInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        acceleration = response.acceleration[0]
        assert torch.isfinite(acceleration).all()
        assert acceleration[0] < 0.0
        assert torch.allclose(acceleration[1], torch.tensor(0.0), atol=1e-6)

    def test_confines_charged_particles_within_radius(self):
        # Arrange
        positions = _ring_positions(radius=3.0)
        velocities = _radial_unit_velocity(positions, speed=0.3)
        interactions = build_interactions(wall="harmonic")

        # Act
        particle_system = _run(steps=300, positions=positions, velocities=velocities, interactions=interactions)

        # Assert
        assert torch.isfinite(particle_system.pos).all()
        assert _max_radius(particle_system.pos) <= R_CONFINEMENT + 1e-2


class TestHardWallEventDriven:
    def test_rescues_escaped_particle_back_to_wall(self):
        # Arrange: partícula fuera de la pared y alejándose (el bug de escape)
        positions = torch.tensor([[7.0, 0.0]])
        velocity = torch.tensor([[1.0, 0.0]])
        decorator = HardWallEventDrivenInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.isfinite(response.positions).all()
        assert torch.isfinite(response.velocity).all()
        assert _max_radius(response.positions) <= R_CONFINEMENT + 1e-5
        assert response.velocity[0, 0] < 0.0  # reflejada hacia adentro

    def test_reflects_particle_crossing_boundary_during_step(self):
        # Arrange: partícula que cruza la pared dentro del paso (t_c < dt)
        positions = torch.tensor([[5.9, 0.0]])
        velocity = torch.tensor([[3.0, 0.0]])
        decorator = HardWallEventDrivenInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.isfinite(response.positions).all()
        assert _max_radius(response.positions) <= R_CONFINEMENT + 1e-5
        assert response.velocity[0, 0] < 0.0  # reflejada hacia adentro

    def test_stays_finite_for_extreme_escaped_position(self):
        # Arrange
        positions = torch.tensor([[100.0, 0.0]])
        velocity = torch.tensor([[50.0, 0.0]])
        decorator = HardWallEventDrivenInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.isfinite(response.positions).all()
        assert torch.isfinite(response.velocity).all()
        assert _max_radius(response.positions) <= R_CONFINEMENT + 1e-5

    def test_never_lets_particles_escape_during_integration(self):
        # Arrange: arranque agresivo sobre la pared con velocidad saliente alta
        positions = _ring_positions(radius=R_CONFINEMENT)
        velocities = _radial_unit_velocity(positions, speed=3.0)
        interactions = build_interactions(wall="hard_wall")

        # Act
        particle_system = _run(steps=300, positions=positions, velocities=velocities, interactions=interactions)

        # Assert
        assert torch.isfinite(particle_system.pos).all()
        assert _max_radius(particle_system.pos) <= R_CONFINEMENT + 1e-2


class TestPairWCA:
    def test_no_force_for_particles_beyond_cutoff(self):
        # Arrange
        positions = torch.tensor([[10.0, 0.0], [-10.0, 0.0]])
        velocity = torch.zeros(2, 2)
        decorator = PairWCAInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.allclose(
            response.acceleration, torch.zeros_like(response.acceleration), atol=1e-6
        )

    def test_repels_overlapping_particles(self):
        # Arrange: dos partículas casi en el mismo punto, separadas en x
        positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        velocity = torch.zeros(2, 2)
        decorator = PairWCAInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert: se alejan entre sí (a0 hacia -x, a1 hacia +x)
        assert torch.isfinite(response.acceleration).all()
        assert response.acceleration[0, 0] < 0.0
        assert response.acceleration[1, 0] > 0.0
        assert torch.allclose(response.acceleration[0, 1], torch.tensor(0.0), atol=1e-6)

    def test_single_particle_has_no_self_force(self):
        # Arrange
        positions = torch.tensor([[0.0, 0.0]])
        velocity = torch.zeros(1, 2)
        decorator = PairWCAInteractionDecorator(NoopInteraction())

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.allclose(
            response.acceleration, torch.zeros_like(response.acceleration), atol=1e-6
        )

    def test_keeps_particles_apart_during_integration(self):
        # Arrange: dos partículas superpuestas en el origen, sin pared
        positions = torch.tensor([[0.0, 0.0], [0.01, 0.0]])
        velocities = torch.zeros(2, 2)
        interactions = PairWCAInteractionDecorator(NoopInteraction())

        # Act
        particle_system = _run(steps=500, positions=positions, velocities=velocities, interactions=interactions)

        # Assert: se separan, siguen finitas y no se vuelven a superponer
        assert torch.isfinite(particle_system.pos).all()
        separation = torch.linalg.norm(
            particle_system.pos[1] - particle_system.pos[0],
        ).item()
        assert separation > 0.1


class TestGravityInteractionDecorator:
    def test_adds_constant_downward_acceleration_independent_of_mass(self):
        # Arrange
        n = 2
        g = 9.8
        base = StubInteraction(acceleration=torch.zeros(n, 2))
        decorator = GravityInteractionDecorator(base)
        positions = torch.tensor([[1.0, 1.0], [-1.0, 2.0]])
        velocity = torch.zeros(n, 2)

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity, sim_props=_sim_props(g=g)))

        # Assert
        expected = torch.tensor([[0.0, -g], [0.0, -g]])
        assert torch.allclose(response.acceleration, expected)

    def test_single_particle_falls_with_acceleration_g(self):
        # Arrange
        base = StubInteraction(acceleration=torch.zeros(1, 2))
        decorator = GravityInteractionDecorator(base)
        positions = torch.tensor([[0.0, 3.0]])
        velocity = torch.zeros(1, 2)

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity, sim_props=_sim_props(g=2.5)))

        # Assert
        assert torch.allclose(response.acceleration, torch.tensor([[0.0, -2.5]]))

    def test_accumulates_on_top_of_wrapped_interaction(self):
        # Arrange
        base_acc = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        base = StubInteraction(acceleration=base_acc)
        decorator = GravityInteractionDecorator(base)
        positions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        velocity = torch.zeros(2, 2)

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity, sim_props=_sim_props(g=5.0)))

        # Assert
        expected = base_acc + torch.tensor([[0.0, -5.0], [0.0, -5.0]])
        assert torch.allclose(response.acceleration, expected)

    def test_zero_gravity_does_not_change_acceleration(self):
        # Arrange
        base_acc = torch.tensor([[2.0, 3.0]])
        base = StubInteraction(acceleration=base_acc)
        decorator = GravityInteractionDecorator(base)
        positions = torch.tensor([[0.0, 0.0]])
        velocity = torch.zeros(1, 2)

        # Act
        response = decorator.compute_aceleration(_query(positions, velocity, sim_props=_sim_props(g=0.0)))

        # Assert
        assert torch.equal(response.acceleration, base_acc)

    def test_preserves_positions_and_velocity_from_wrapped_interaction(self):
        # Arrange
        pos = torch.tensor([[7.0, 0.0]])
        vel = torch.tensor([[1.0, 0.0]])
        base = StubInteraction(acceleration=torch.zeros(1, 2), positions=pos, velocity=vel)
        decorator = GravityInteractionDecorator(base)

        # Act
        response = decorator.compute_aceleration(_query(pos, vel, sim_props=_sim_props(g=2.0)))

        # Assert
        assert torch.equal(response.positions, pos)
        assert torch.equal(response.velocity, vel)

    def test_charged_particles_drift_downward_under_gravity(self):
        # Arrange
        g = 2.0
        positions = _ring_positions(radius=3.0)
        velocities = torch.zeros_like(positions)
        interactions = build_interactions(add_gravity=True)

        # Act
        particle_system = _run(
            steps=200,
            positions=positions,
            velocities=velocities,
            interactions=interactions,
            sim_props=_sim_props(g=g),
        )

        # Assert
        assert torch.isfinite(particle_system.pos).all()
        assert float(particle_system.pos[:, 1].mean()) < 0.0  # cayeron hacia -y


@pytest.mark.parametrize(
    "wall",
    [member for member in ConfinementType],
)
class TestBuildInteractions:
    def test_wall_variant_produces_finite_response(self, wall):
        # Arrange
        positions = torch.tensor([[2.0, 0.0], [-1.0, 1.0]])
        velocity = torch.zeros(2, 2)

        # Act
        response = build_interactions(wall=wall).compute_aceleration(_query(positions, velocity))

        # Assert
        assert torch.isfinite(response.acceleration).all()
        if response.positions is not None:
            assert torch.isfinite(response.positions).all()
        if response.velocity is not None:
            assert torch.isfinite(response.velocity).all()
