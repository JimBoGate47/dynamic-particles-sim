from src.common.domain.models.particle import ParticleORM
from src.common.domain.types.particle import Particle2D, Particle, Position2D, Velocity2D, Aceleration2D
from src.common.domain.types.properties import PhysicalProperties


def build_particle_2d(
        orm_instance: ParticleORM
) -> Particle:
    return Particle2D(
        r=Position2D(
            x=orm_instance.r[0],
            y=orm_instance.r[1],
        ),
        v=Velocity2D(
            x=0,
            y=0,
        ),
        a=Aceleration2D(
            x=0,
            y=0,
        ),
        phys_props=PhysicalProperties(
            q=orm_instance.phys_props.get("q", -999),
            m=orm_instance.phys_props.get("m", -999),
        ),
    )
