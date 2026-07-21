from typing import Union

from beanie import Link

from src.common.domain.entities import Constants
from src.common.domain.entities.properties import SimulationProps
from src.common.domain.models.constants import ConstantsORM


def build_constants(constants: Union[ConstantsORM, Link[ConstantsORM]]) -> Constants:
    return Constants(
        id=str(constants.id),
        name=constants.name,
        sim_props=SimulationProps(
            g=constants.g,
            k=constants.k,
            dt=constants.dt,
            min_vel=constants.min_vel,
            r_confinement=constants.r_confinement,
            k_confinement=0,
            beta=0,
        ),
        friction=constants.friction,
        confinement=constants.confinement,
        ruta=constants.ruta,
        version=constants.version,
        barra_height=constants.barra_height,
        barra_qlamb=constants.barra_qlamb,
    )
