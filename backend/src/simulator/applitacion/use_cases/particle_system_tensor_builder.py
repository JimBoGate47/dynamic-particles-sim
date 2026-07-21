from dataclasses import dataclass

from src.common.domain.entities import Snapshot
from src.common.domain.entities.properties import SimulationProps
from src.simulator.applitacion.mixins import SnapshotFinderMixin
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.infrastructure.builders.system_tensor import build_system_tensor
from src.common.domain.interfaces import UseCase


@dataclass
class ParticleSystemTensorBuilderById(UseCase, SnapshotFinderMixin):
    sim_props: SimulationProps
    fetch_links: bool = False

    async def execute(self, *args, **kwargs) -> ParticleSystem2DTensor:
        snapshot: Snapshot = await self.find_by_id()
        if not snapshot:
            raise ValueError("Not snapshot found")
        system_tensor = build_system_tensor(snapshot)
        return ParticleSystem2DTensor(
            pos=system_tensor.pos,
            vel=system_tensor.vel,
            acc=system_tensor.acc,
            phys_props=system_tensor.phys_props,
        )
