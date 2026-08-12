from dataclasses import dataclass, field
from typing import Awaitable, Callable

from loguru import logger

from src.common.domain.entities import Snapshot
from src.common.domain.enums import ConfinementType
from src.common.domain.events import (
    SimulationSnapshotPersisted,
    SimulationStepCompleted,
)
from src.simulator.infrastructure.interaction import build_interactions
from src.simulator.infrastructure.metrics.engine import MetricsEngine, SamplingPolicy


@dataclass
class MetricsLoggingHandler:
    policy: SamplingPolicy = field(default_factory=SamplingPolicy)
    wall: ConfinementType = ConfinementType.HARMONIC
    add_gravity: bool = False
    snapshot_loader: Callable[[str], Awaitable[Snapshot]] | None = None

    async def on_step(self, event: SimulationStepCompleted) -> None:
        if not self.policy.should_capture(event.step_ordinal, event.total_steps):
            return
        metrics = MetricsEngine.compute_from_response(
            step=event.step,
            positions=event.positions,
            velocity=event.velocity,
            phys_props=event.phys_props,
            sim_props=event.sim_props,
            response=event.response,
        )
        logger.info("metrics step={} data={}", metrics.step, metrics.to_dict())

    async def on_snapshot(self, event: SimulationSnapshotPersisted) -> None:
        if self.snapshot_loader is None:
            logger.warning(
                "Snapshot event received but no snapshot_loader configured"
            )
            return
        snapshot = await self.snapshot_loader(event.snapshot_id)
        if snapshot is None:
            logger.warning("Snapshot not found for metrics: {}", event.snapshot_id)
            return
        interactions = build_interactions(
            add_gravity=self.add_gravity,
            wall=self.wall,
        )
        metrics = MetricsEngine.compute(snapshot, interactions)
        logger.info(
            "metrics snapshot={} data={}", metrics.step, metrics.to_dict()
        )
