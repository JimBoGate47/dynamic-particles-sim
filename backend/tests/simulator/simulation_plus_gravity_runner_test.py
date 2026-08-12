import pytest

from src.common.domain.entities.properties import SimulationProps
from src.simulator.applitacion.use_cases.simulation_plus_gravity_runner import (
    SimulationPlusGravityRunner,
)


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


def _runner(n_steps: int) -> SimulationPlusGravityRunner:
    return SimulationPlusGravityRunner(
        snapshot_id="snapshot-id",
        orm_snapshot=None,
        n_steps=n_steps,
    )


class TestUpdatedGravitySimProps:
    def test_generates_multiples_of_delta_gravity(self):
        # Arrange
        runner = _runner(n_steps=3)

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(g=9.0, delta_gravity=0.5))

        # Assert: g_i = i * delta_gravity, empezando desde delta_gravity (sin sumar sobre g)
        assert [props.g for props in props_list] == [0.5, 1.0, 1.5]
        assert props_list[0].g == 0.5

    def test_delta_gravity_is_the_spacing_between_values(self):
        # Arrange
        runner = _runner(n_steps=5)

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(g=9.0, delta_gravity=0.25))

        # Assert
        values = [props.g for props in props_list]
        assert values == [0.25, 0.5, 0.75, 1.0, 1.25]
        assert all(
            values[i + 1] - values[i] == pytest.approx(0.25)
            for i in range(len(values) - 1)
        )

    def test_final_value_is_open_and_unrestricted(self):
        # Arrange
        runner = _runner(n_steps=4)

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props(g=9.0, delta_gravity=2.0))

        # Assert: el final queda abierto (n_steps * delta), sin tope
        assert [props.g for props in props_list] == [2.0, 4.0, 6.0, 8.0]
        assert props_list[-1].g > props_list[0].g

    def test_returns_distinct_objects(self):
        # Arrange
        runner = _runner(n_steps=3)

        # Act
        props_list = runner.updated_gravity_sim_props(_sim_props())

        # Assert: cada fase es una copia, no la misma instancia mutada
        assert len({id(props) for props in props_list}) == 3
        assert all(props is not props_list[0] for props in props_list[1:])

    def test_original_sim_props_is_not_mutated(self):
        # Arrange
        runner = _runner(n_steps=2)
        sim_props = _sim_props(g=9.0, delta_gravity=2.0)

        # Act
        props_list = runner.updated_gravity_sim_props(sim_props)

        # Assert: el original conserva su g y no participa en la secuencia
        assert sim_props.g == 9.0
        assert [props.g for props in props_list] == [2.0, 4.0]

    def test_raises_without_delta_gravity(self):
        # Arrange
        runner = _runner(n_steps=1)

        # Act / Assert
        try:
            runner.updated_gravity_sim_props(_sim_props(delta_gravity=None))
        except Exception as exc:
            assert "Delta gravity not found" in str(exc)
        else:
            raise AssertionError("Expected Exception was not raised")
