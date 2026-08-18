import pytest
import torch

from src.common.domain.entities.constants import Constants
from src.common.domain.entities.particle import Particle
from src.common.domain.entities.properties import PhysicalProps, SimulationProps
from src.common.domain.entities.snapshot import Snapshot
from src.simulator.infrastructure.helpers.interaction import build_query_from_snapshot
from src.simulator.infrastructure.interaction import (
    HardWallEventDrivenInteractionDecorator,
    build_interactions,
)
from src.simulator.infrastructure.metrics.engine import (
    MetricsEngine,
    MetricsSamplingMode,
    SamplingPolicy,
    kinetic_energy,
    compute_metrics_for_snapshots,
)
from src.simulator.infrastructure.queries import (
    GenericInteractionQuery,
    GenericInteractionResponse,
)

R_CONFINEMENT = 6.0
K_CONFINEMENT = 0.5
DT = 0.1


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


def _response_for(query) -> GenericInteractionResponse:
    return build_interactions(wall="harmonic").compute_aceleration(query)


def _build_snapshot(step: int = 5, n: int = 2) -> Snapshot:
    particles = [
        Particle(
            r=[1.0, 0.0],
            v=[0.1, 0.2],
            a=[0.0, 0.0],
            phys_props={"q": 1.0, "m": 1.0},
        ),
        Particle(
            r=[0.0, 1.0],
            v=[-0.1, 0.0],
            a=[0.0, 0.0],
            phys_props={"q": 1.0, "m": 1.0},
        ),
    ][:n]
    return Snapshot(
        id="snapshot-id",
        step=step,
        constants=Constants(name="test", sim_props=_sim_props()),
        particles=particles,
    )


class TestForceContributions:
    def test_build_interactions_records_all_components(self):
        # Arrange
        positions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        velocity = torch.zeros(2, 2)

        # Act
        response = _response_for(_query(positions, velocity))

        # Assert
        assert response.contributions is not None
        assert {"electrostatic", "harmonic_wall", "friction"} <= set(response.contributions)
        for force in response.contributions.values():
            assert force.shape == (2, 2)

    def test_sum_of_force_contributions_matches_total(self):
        # Arrange
        positions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        velocity = torch.tensor([[1.0, -1.0], [0.5, 0.2]])
        query = _query(positions, velocity)

        # Act
        response = _response_for(query)

        # Assert: Σ F_i = m * a_total
        total_force = sum(response.contributions.values())
        assert torch.allclose(
            total_force,
            query.phys_props.m * response.acceleration,
            atol=1e-6,
        )

    def test_friction_force_is_negative_beta_times_mass_times_velocity(self):
        # Arrange
        positions = torch.tensor([[1.0, 0.0]])
        velocity = torch.tensor([[2.0, -3.0]])
        query = _query(positions, velocity)

        # Act
        response = _response_for(query)

        # Assert
        expected = -query.sim_props.beta * query.phys_props.m * velocity
        assert torch.allclose(response.contributions["friction"], expected, atol=1e-6)

    def test_gravity_contribution_is_downward_force(self):
        # Arrange
        g = 9.8
        positions = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        velocity = torch.zeros(2, 2)
        query = _query(positions, velocity, sim_props=_sim_props(g=g))

        # Act
        response = build_interactions(add_gravity=True, wall="harmonic").compute_aceleration(query)

        # Assert: F_grav = m * (0, -g) = (0, -g) con m=1
        expected = torch.zeros(2, 2)
        expected[:, 1] = -g
        assert torch.allclose(response.contributions["gravity"], expected, atol=1e-6)

    def test_hard_wall_overrides_state_without_adding_force(self):
        # Arrange: partícula fuera de la pared alejándose
        positions = torch.tensor([[7.0, 0.0]])
        velocity = torch.tensor([[1.0, 0.0]])
        query = _query(positions, velocity)

        # Act
        response = HardWallEventDrivenInteractionDecorator(
            build_interactions(wall="harmonic")
        ).compute_aceleration(query)

        # Assert: la pared dura no registra fuerza y sobreescribe pos/vel
        assert "hard_wall" not in response.contributions
        assert torch.linalg.norm(response.positions, dim=1).max() <= R_CONFINEMENT + 1e-5
        assert response.velocity[0, 0] < 0.0


class TestSamplingPolicy:
    def test_final_only_captures_last_step(self):
        # Arrange
        policy = SamplingPolicy(mode=MetricsSamplingMode.FINAL_ONLY)

        # Assert
        for step in range(1, 10):
            assert not policy.should_capture(step, 10)
        assert policy.should_capture(10, 10)

    def test_all_captures_every_step(self):
        # Arrange
        policy = SamplingPolicy(mode=MetricsSamplingMode.ALL)

        # Assert
        assert all(policy.should_capture(step, 10) for step in range(1, 11))

    def test_every_n_captures_multiplos_and_final(self):
        # Arrange
        policy = SamplingPolicy(mode=MetricsSamplingMode.EVERY_N, every_n=3)

        # Assert
        assert policy.should_capture(3, 10)
        assert policy.should_capture(6, 10)
        assert policy.should_capture(10, 10)
        assert not policy.should_capture(1, 10)
        assert not policy.should_capture(2, 10)


class TestMetricsEngine:
    def test_compute_is_deterministic_and_reproducible(self):
        # Arrange
        snapshot = _build_snapshot()

        # Act
        m1 = MetricsEngine.compute(snapshot, build_interactions(wall="harmonic"))
        m2 = MetricsEngine.compute(snapshot, build_interactions(wall="harmonic"))

        # Assert
        assert m1.step == m2.step == snapshot.step
        assert set(m1.forces) == set(m2.forces)
        for name in m1.forces:
            assert torch.allclose(m1.forces[name], m2.forces[name])
        assert m1.aggregates == m2.aggregates

    def test_metrics_include_aggregates(self):
        # Arrange
        snapshot = _build_snapshot()

        # Act
        metrics = MetricsEngine.compute(snapshot, build_interactions(wall="harmonic"))

        # Assert
        for key in (
                "mean_speed",
                "rms_speed",
                "kinetic_energy",
                "temperature",
                "coulomb_energy",
                "x.min_distance",
                "x.max_distance",
                "y.min_distance",
                "y.max_distance",
                "electrostatic.mean_force",
                "harmonic_wall.mean_force",
                "friction.mean_force",
        ):
            assert key in metrics.aggregates

    def test_forces_are_per_particle_magnitudes(self):
        # Arrange
        n = 2
        snapshot = _build_snapshot(n=n)
        interactions = build_interactions(wall="harmonic")
        response = interactions.compute_aceleration(
            build_query_from_snapshot(snapshot)
        )

        # Act
        metrics = MetricsEngine.compute(snapshot, interactions)

        # Assert: fuerzas como módulos por partícula (shape [N], >= 0)
        assert set(metrics.forces) == set(response.contributions)
        for name, force in response.contributions.items():
            assert metrics.forces[name].shape == (n,)
            assert torch.allclose(
                metrics.forces[name], torch.linalg.norm(force, dim=1)
            )
            assert bool(torch.all(metrics.forces[name] >= 0))

    def test_to_dict_is_serializable(self):
        # Arrange
        snapshot = _build_snapshot()
        metrics = MetricsEngine.compute(snapshot, build_interactions(wall="harmonic"))

        # Act
        payload = metrics.to_dict(include_forces=True)

        # Assert
        assert isinstance(payload["step"], int)
        assert all(isinstance(v, list) for v in payload["forces"].values())
        assert all(
            all(isinstance(x, float) for x in magnitudes)
            for magnitudes in payload["forces"].values()
        )
        assert all(isinstance(v, float) for v in payload["aggregates"].values())

    def test_to_dict_hides_forces_by_default(self):
        # Arrange
        snapshot = _build_snapshot()
        metrics = MetricsEngine.compute(snapshot, build_interactions(wall="harmonic"))

        # Act
        payload = metrics.to_dict()

        # Assert: por defecto solo lo principal, sin fuerzas
        assert set(payload) == {"step", "aggregates"}
        assert "forces" not in payload
        assert all(isinstance(v, float) for v in payload["aggregates"].values())

    def test_query_uses_gravity_from_snapshot_metadata(self):
        # Arrange: constants con g=9.0, pero metadata indica otra gravedad
        snapshot = _build_snapshot()
        snapshot.metadata = {"g": 4.5}

        # Act
        query = build_query_from_snapshot(snapshot)

        # Assert
        assert query.sim_props.g == 4.5

    def test_query_falls_back_to_constants_without_metadata(self):
        # Arrange
        snapshot = _build_snapshot()

        # Act
        query = build_query_from_snapshot(snapshot)

        # Assert
        assert query.sim_props.g == snapshot.constants.sim_props.g

    def test_compute_uses_gravity_from_metadata(self):
        # Arrange
        snapshot = _build_snapshot()
        snapshot.metadata = {"g": 4.5}
        interactions = build_interactions(add_gravity=True, wall="harmonic")

        # Act
        metrics = MetricsEngine.compute(snapshot, interactions)

        # Assert: fuerza de gravedad = m * g_metadata (m=1)
        assert "gravity" in metrics.forces
        assert torch.allclose(metrics.forces["gravity"], torch.full((2,), 4.5))


class TestComputeMetricsForSnapshots:
    def test_samples_according_to_policy(self):
        # Arrange
        snapshots = [_build_snapshot(step=step) for step in (1, 2, 3)]

        # Act
        metrics = compute_metrics_for_snapshots(
            snapshots=snapshots,
            interactions=build_interactions(wall="harmonic"),
            policy=SamplingPolicy(mode=MetricsSamplingMode.ALL),
        )

        # Assert
        assert [m.step for m in metrics] == [1, 2, 3]

    def test_final_only_returns_single_last_sample(self):
        # Arrange
        snapshots = [_build_snapshot(step=step) for step in (1, 2, 3)]

        # Act
        metrics = compute_metrics_for_snapshots(
            snapshots=snapshots,
            interactions=build_interactions(wall="harmonic"),
            policy=SamplingPolicy(mode=MetricsSamplingMode.FINAL_ONLY),
        )

        # Assert
        assert [m.step for m in metrics] == [3]


class TestMetricsEngineComputeFromResponse:
    def test_uses_response_contributions_without_rerunning_chain(self):
        # Arrange
        snapshot = _build_snapshot()
        interactions = build_interactions(wall="harmonic")
        query = build_query_from_snapshot(snapshot)
        response = interactions.compute_aceleration(query)

        # Act
        metrics = MetricsEngine.compute_from_response(
            step=snapshot.step,
            positions=query.positions,
            velocity=query.velocity,
            phys_props=query.phys_props,
            sim_props=query.sim_props,
            response=response,
        )

        # Assert: reusa el response (no re-ejecuta la cadena)
        assert metrics.step == snapshot.step
        assert set(metrics.forces) == set(response.contributions)
        for name, force in response.contributions.items():
            expected = torch.linalg.norm(force, dim=1)
            assert torch.allclose(metrics.forces[name], expected)
        assert "kinetic_energy" in metrics.aggregates

    def test_matches_compute_for_same_state(self):
        # Arrange
        snapshot = _build_snapshot()
        interactions = build_interactions(wall="harmonic")
        query = build_query_from_snapshot(snapshot)
        response = interactions.compute_aceleration(query)

        # Act
        m1 = MetricsEngine.compute(snapshot, interactions)
        m2 = MetricsEngine.compute_from_response(
            step=snapshot.step,
            positions=query.positions,
            velocity=query.velocity,
            phys_props=query.phys_props,
            sim_props=query.sim_props,
            response=response,
        )

        # Assert
        assert m1.step == m2.step
        assert set(m1.forces) == set(m2.forces)
        for name in m1.forces:
            assert torch.allclose(m1.forces[name], m2.forces[name])
        assert m1.aggregates == m2.aggregates


class TestAggregates:
    def test_kinetic_energy(self):
        # Arrange
        m = torch.ones(1, 1) * 2.0
        v = torch.tensor([[1.0, 0.0]])

        # Act / Assert: KE = 0.5 * 2 * 1^2 = 1.0
        assert kinetic_energy(m, v) == pytest.approx(1.0)
