from typing import Union

from beanie import Link

from backend.src.common.domain.models.constants import ConstantsORM
from backend.src.common.domain.entities import Constants


def build_constants(constants: Union[ConstantsORM, Link[ConstantsORM]]) -> Constants:
    return Constants(
        id=str(constants.id),
        name=constants.name,
        g=constants.g,
        k=constants.k,
        min_vel=constants.min_vel,
        friction=constants.friction,
        ruta=constants.ruta,
        dt=constants.dt,
        confinement=constants.confinement,
        r_confinement=constants.r_confinement,
        version=constants.version,
        barra_height=constants.barra_height,
        barra_qlamb=constants.barra_qlamb,
    )
