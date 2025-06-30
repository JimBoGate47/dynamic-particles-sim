from backend.src.common.domain.entities import Constants
from backend.src.common.domain.types.properties import SimulationProperties


def build_constants(
        sim_props: SimulationProperties,
        name: str = "nombre2",
) -> Constants:
    return Constants(
        name=name,
        g=sim_props.g,
        k=sim_props.k,
        dt=sim_props.dt,
        min_vel=sim_props.min_vel,
    )
