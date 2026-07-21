from typing import Union

from beanie import Link

from src.common.domain.entities import Constants
from src.common.domain.models.constants import ConstantsORM


def build_constants(constants: Union[ConstantsORM, Link[ConstantsORM]]) -> Constants:
    return Constants(
        id=str(constants.id),
        name=constants.name,
        sim_props=constants.sim_props,
        friction=constants.friction,
        confinement=constants.confinement,
        ruta=constants.ruta,
        version=constants.version,
        barra_height=constants.barra_height,
        barra_qlamb=constants.barra_qlamb,
    )
