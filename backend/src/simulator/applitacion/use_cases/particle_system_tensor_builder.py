from dataclasses import dataclass

from backend.src.common.domain.entities import Snapshot
from backend.src.common.domain.entities.properties import SimulationProps
from backend.src.simulator.applitacion.mixins import SnapshotFinderMixin
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from backend.src.simulator.infrastructure.builders.system_tensor import build_system_tensor
from src.common.use_cases import UseCase


@dataclass
class ParticleSystemTensorBuilderById(UseCase, SnapshotFinderMixin):
    sim_props: SimulationProps

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
            sim_props=self.sim_props,
        )
