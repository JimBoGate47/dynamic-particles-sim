import pytest

from src.common.domain.entities.properties import SimulationProps
from src.simulator.applitacion.use_cases.simulation_plus_gravity_runner import (
    SimulationPlusGravityRunner,
)
from src.simulator.domain.entities.gravity import GravityConfig


def _sim_props(**overrides) -> SimulationProps:
    defaults = dict(
        g=9.0,
        k=10.0,
        min_vel=0.0,
        r_confinement=6.0,
        k_confinement=0.5,
        beta=0.6,
        dt=0.1,
        delta_gravity=0.5,
    )
    defaults.update(overrides)
    return SimulationProps(**defaults)


def _runner(gravity_config: GravityConfig) -> SimulationPlusGravityRunner:
    return SimulationPlusGravityRunner(
        snapshot_id="snapshot-id",
        orm_snapshot=None,
        gravity_config=gravity_config,
    )


class TestUpdatedGravitySimProps:
    def test_generates_sequence_with_delta_g_as_factor(self):
        # Arrange
        runner = _runner(GravityConfig(start=1, end=3, delta_g=0.5))

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(g=9.0))

        # Assert: números ascendentes start..end multiplicados por delta_g
        assert [props.g for props in props_list] == [0.5, 1.0, 1.5]
        assert props_list[0].g == 0.5

    def test_delta_g_is_the_multiplicative_factor(self):
        # Arrange
        runner = _runner(GravityConfig(start=1, end=5, delta_g=0.25))

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(g=9.0))

        # Assert
        values = [props.g for props in props_list]
        assert values == [0.25, 0.5, 0.75, 1.0, 1.25]
        assert all(
            values[i + 1] - values[i] == pytest.approx(0.25)
            for i in range(len(values) - 1)
        )

    def test_end_defines_number_of_steps(self):
        # Arrange
        runner = _runner(GravityConfig(start=1, end=4, delta_g=2.0))

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(g=9.0))

        # Assert: end acota el rango ascendente (end - start + 1 valores)
        assert [props.g for props in props_list] == [2.0, 4.0, 6.0, 8.0]
        assert len(props_list) == 4

    def test_returns_distinct_objects(self):
        # Arrange
        runner = _runner(GravityConfig(start=1, end=3, delta_g=1.0))

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props())

        # Assert: cada fase es una copia, no la misma instancia mutada
        assert len({id(props) for props in props_list}) == 3
        assert all(props is not props_list[0] for props in props_list[1:])

    def test_original_sim_props_is_not_mutated(self):
        # Arrange
        runner = _runner(GravityConfig(start=1, end=2, delta_g=2.0))
        sim_props = _sim_props(g=9.0)

        # Act
        props_list = runner.updated_gravity_sim_props(sim_props)

        # Assert: el original conserva su g y no participa en la secuencia
        assert sim_props.g == 9.0
        assert [props.g for props in props_list] == [2.0, 4.0]

    def test_delta_gravity_on_sim_props_is_ignored(self):
        # Arrange
        runner = _runner(GravityConfig(start=1, end=2, delta_g=1.0))

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(delta_gravity=None))

        # Assert: ya no depende de sim_props.delta_gravity
        assert [props.g for props in props_list] == [1.0, 2.0]