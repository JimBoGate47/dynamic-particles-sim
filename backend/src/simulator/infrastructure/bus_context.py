from typing import Awaitable, Callable

from src.common.domain.entities import Snapshot
from src.common.domain.enums import ConfinementType
from src.common.domain.events import (
    SimulationSnapshotPersisted,
    SimulationStepCompleted,
)
from src.common.domain.interfaces import EventBus
from src.simulator.infrastructure.event_bus import AnyIOEventBus
from src.simulator.infrastructure.metrics.engine import MetricsSamplingMode, SamplingPolicy
from src.simulator.infrastructure.metrics.logging import MetricsLoggingHandler


def build_bus_context(
    *,
    wall: ConfinementType,
    add_gravity: bool,
    metrics_mode: MetricsSamplingMode,
    metrics_every_n: int,
    snapshot_loader: Callable[[str], Awaitable[Snapshot]] | None = None,
) -> EventBus:
    """Crea el EventBus de la simulación y registra sus handlers.

    Los handlers son inyectables; por ahora se registra el de métricas, pero la
    composición queda preparada para agregar más sin acoplar el bus a uno solo.
    """
    bus: EventBus = AnyIOEventBus()
    handler = MetricsLoggingHandler(
        policy=SamplingPolicy(mode=metrics_mode, every_n=metrics_every_n),
        wall=wall,
        add_gravity=add_gravity,
        snapshot_loader=snapshot_loader,
    )
    bus.subscribe(SimulationStepCompleted, handler.on_step)
    bus.subscribe(SimulationSnapshotPersisted, handler.on_snapshot)
    return bus
