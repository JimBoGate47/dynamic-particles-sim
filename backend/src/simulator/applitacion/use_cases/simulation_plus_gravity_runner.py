from dataclasses import dataclass

from src.common.domain.entities import Snapshot
from src.common.domain.enums import ConfinementType
from src.common.domain.entities.properties import SimulationProps
from src.common.domain.interfaces import UseCase
from src.simulator.applitacion.mixins import SnapshotFinderMixin
from src.simulator.applitacion.use_cases.simulation_mixins import SimulationStabilizerMixin
from src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from src.simulator.infrastructure.builders.system_tensor import build_system_tensor
from src.simulator.infrastructure.interaction import (
    build_interactions,
)

_SAVE_AT_MOD_DISABLED = 0


@dataclass
class SimulationPlusGravityRunner(UseCase, SnapshotFinderMixin, SimulationStabilizerMixin):
    stabilization_steps: int = 506
    n_steps: int = 10
    fetch_links: bool = True
    wall: ConfinementType = ConfinementType.HARMONIC

    async def execute(self, *args, **kwargs) -> list[Snapshot]:
        snapshot = await self.find_by_id()
        if not snapshot:
            raise ValueError("Snapshot not found")
        if not snapshot.constants:
            raise ValueError("Snapshot has no linked constants")

        interactions = build_interactions(add_gravity=True, wall=self.wall)

        system_tensor = build_system_tensor(snapshot)
        ps = ParticleSystem2DTensor(
            pos=system_tensor.pos,
            vel=system_tensor.vel,
            acc=system_tensor.acc,
            phys_props=system_tensor.phys_props,
            step=snapshot.step,
        )

        snapshots: list[Snapshot] = []
        for sim_props in self.updated_gravity_sim_props(
                sim_props=snapshot.constants.sim_props,
        ):
            stabilized_snapshots: list[Snapshot] = await self.stabilize(
                ps=ps,
                constants_id=snapshot.constants.id_object,
                batch_id=snapshot.batch_id,
                sim_props=sim_props,
                interactions=interactions,
                stabilization_steps=self.stabilization_steps,
                save_at_mod=_SAVE_AT_MOD_DISABLED,
            )
            snapshots.append(stabilized_snapshots[-1])
        return snapshots

    def updated_gravity_sim_props(self, sim_props: SimulationProps) -> list[SimulationProps]:
        sim_props_list: list[SimulationProps] = []
        if not sim_props.delta_gravity_exists:
            raise Exception("Delta gravity not found")

        for i in range(1, self.n_steps + 1):
            sim_props.model_copy(
                update={"g": i * sim_props.delta_gravity},
            )
            sim_props_list.append(sim_props)
        return sim_props_list
