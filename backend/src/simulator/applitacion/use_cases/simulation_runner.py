from dataclasses import dataclass

from loguru import logger

from src.common.domain.entities import Snapshot
from src.common.domain.interfaces import UseCase
from src.simulator.applitacion.mixins import SnapshotFinderMixin
from src.simulator.applitacion.use_cases.simulation_mixins import SimulationStabilizerMixin
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.infrastructure.builders.system_tensor import build_system_tensor
from src.simulator.infrastructure.interaction import (
    build_interactions,
)


@dataclass
class SimulationStabilizerRunner(UseCase, SnapshotFinderMixin, SimulationStabilizerMixin):
    n_steps: int = 506
    save_at_mod: int = 100
    fetch_links: bool = True

    async def execute(self, *args, **kwargs) -> list[Snapshot]:
        snapshot = await self.find_by_id()
        if not snapshot:
            raise ValueError("Snapshot not found")
        if not snapshot.constants:
            raise ValueError("Snapshot has no linked constants")

        interactions = build_interactions(add_gravity=False)

        system_tensor = build_system_tensor(snapshot)
        ps = ParticleSystem2DTensor(
            pos=system_tensor.pos,
            vel=system_tensor.vel,
            acc=system_tensor.acc,
            phys_props=system_tensor.phys_props,
            step=snapshot.step,
        )

        logger.info("Starting simulation: step={} batch_id={} sim_props={}",
                    ps.step, snapshot.batch_id, snapshot.constants.sim_props)
        snapshots = await self.stabilize(
            ps=ps,
            constants_id=snapshot.constants.id_object,
            batch_id=snapshot.batch_id,
            sim_props=snapshot.constants.sim_props,
            interactions=interactions,
            stabilization_steps=self.n_steps,
            save_at_mod=self.save_at_mod,
        )
        return snapshots
