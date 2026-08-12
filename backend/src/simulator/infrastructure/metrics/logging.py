from dataclasses import dataclass, field
import json
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


def _format_payload(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=False)


@dataclass
class MetricsLoggingHandler:
    policy: SamplingPolicy = field(default_factory=SamplingPolicy)
    wall: ConfinementType = ConfinementType.HARMONIC
    add_gravity: bool = False
    include_forces: bool = False
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
        logger.info(
            "metrics step={} data=\n{}",
            metrics.step,
            _format_payload(metrics.to_dict(include_forces=self.include_forces)),
        )

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
            "metrics snapshot={} data=\n{}",
            metrics.step,
            _format_payload(metrics.to_dict(include_forces=self.include_forces)),
        )
