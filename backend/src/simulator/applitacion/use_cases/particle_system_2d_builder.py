from dataclasses import dataclass

from src.common.domain.entities.particle_system import ParticleSystem2D
from src.common.domain.entities.properties import PhysicalProps
from src.common.domain.interfaces import UseCase
from src.simulator.domain.constants import DeviceType, DEVICE_MAP
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.infrastructure.builders.particle_system import build_particles_2d


@dataclass
class ParticleSystem2DBuilder(UseCase):
    n_particles: int
    R: float
    device: DeviceType
    charges: list[float] | None = None

    async def execute(self, *args, **kwargs) -> ParticleSystem2D:
        device = DEVICE_MAP[self.device]
        charges = self.charges or [1.0]
        phys_props = PhysicalProps.from_charges(
            n_particles=self.n_particles,
            charges=charges,
            device=device,
        )
        pos = ParticleSystem2DTensor.initialize_particles_in_circle(
            n_particles=self.n_particles,
            R=self.R,
            device=device,
        )
        ps_tensor = ParticleSystem2DTensor(
            pos=pos,
            phys_props=phys_props,
        )
        return build_particles_2d(ps_tensor)
