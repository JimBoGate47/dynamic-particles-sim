from dataclasses import dataclass

from src.common.domain.enums import ConfinementType
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository
from src.simulator.infrastructure.interaction import build_interactions
from src.simulator.infrastructure.metrics.engine import (
    MetricsSamplingMode,
    SamplingPolicy,
    SimulationMetrics,
    compute_metrics_for_snapshots,
)


@dataclass
class SimulationMetricsRunner(UseCase):
    batch_id: str
    orm_snapshot: SnapshotRepository
    wall: ConfinementType = ConfinementType.HARMONIC
    add_gravity: bool = False
    metrics_mode: MetricsSamplingMode = MetricsSamplingMode.FINAL_ONLY
    metrics_every_n: int = 1

    async def execute(self, *args, **kwargs) -> list[SimulationMetrics]:
        snapshots = await self.orm_snapshot.filter(
            params=SnapshotsFilter(batch_id=self.batch_id)
        )
        if not snapshots:
            return []

        interactions = build_interactions(add_gravity=self.add_gravity, wall=self.wall)
        return compute_metrics_for_snapshots(
            snapshots=snapshots,
            interactions=interactions,
            policy=SamplingPolicy(
                mode=self.metrics_mode,
                every_n=self.metrics_every_n,
            ),
        )
