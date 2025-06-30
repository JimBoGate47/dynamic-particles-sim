from backend.src.common.domain.entities import Snapshot
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2D
from backend.src.simulator.infrastructure.builders.constants import build_constants
from backend.src.simulator.infrastructure.builders.particle import build_particles


# TODO tal vez innecesario
def build_snapshot(
        particles: ParticleSystem2D,
        name: str = "nombre1",
) -> Snapshot:
    return Snapshot(
        particles=build_particles(
            mx_particles=particles,
            phys_props=particles.phys_props
        ),
        constants=build_constants(sim_props=particles.sim_props, name=name),
        step=particles.step,
    )
