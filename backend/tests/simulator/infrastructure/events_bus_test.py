import asyncio
import time
from dataclasses import dataclass

import torch
from bson import ObjectId

from src.common.domain.entities.properties import PhysicalProps, SimulationProps
from src.common.domain.events import (
    DomainEvent,
    SimulationSnapshotPersisted,
    SimulationStepCompleted,
)
from src.simulator.infrastructure.event_bus import AnyIOEventBus
from src.simulator.infrastructure.interaction import build_interactions
from src.simulator.infrastructure.metrics.engine import MetricsSamplingMode, SamplingPolicy
from src.simulator.infrastructure.metrics.logging import MetricsLoggingHandler
from src.simulator.infrastructure.queries import GenericInteractionQuery


@dataclass
class _Foo(DomainEvent):
    value: int


@dataclass
class _Bar(DomainEvent):
    value: int


def _run(coro):
    return asyncio.run(coro)


def _sim_props() -> SimulationProps:
    return SimulationProps(
        g=0.0,
        k=10.0,
        min_vel=0.0,
        r_confinement=6.0,
        k_confinement=0.5,
        beta=0.6,
        dt=0.1,
    )


def _step_event(*, step_ordinal, total_steps=5) -> SimulationStepCompleted:
    positions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    velocity = torch.zeros(2, 2)
    sim_props = _sim_props()
    phys_props = PhysicalProps(
        q=torch.full((2, 1), 1.0),
        m=torch.ones(2, 1),
    )
    response = build_interactions(wall="harmonic").compute_aceleration(
        GenericInteractionQuery(
            positions=positions,
            velocity=velocity,
            sim_props=sim_props,
            phys_props=phys_props,
        )
    )
    return SimulationStepCompleted(
        step=step_ordinal,
        step_ordinal=step_ordinal,
        total_steps=total_steps,
        batch_id="batch",
        constants_id=ObjectId(),
        positions=positions,
        velocity=velocity,
        phys_props=phys_props,
        sim_props=sim_props,
        response=response,
    )


class _FakeLogger:
    def __init__(self):
        self.info_calls = []
        self.warning_calls = []

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


class TestEventBus:
    def test_publish_dispatches_to_subscribed_handler(self):
        async def scenario():
            bus = AnyIOEventBus()
            received = []

            async def handler(event):
                received.append(event.value)

            bus.subscribe(_Foo, handler)
            await bus.start()
            bus.publish(_Foo(value=1))
            await bus.stop()
            return received

        assert _run(scenario()) == [1]

    def test_multiple_handlers_same_event_are_called(self):
        async def scenario():
            bus = AnyIOEventBus()
            calls = []

            async def h1(event):
                calls.append(("h1", event.value))

            async def h2(event):
                calls.append(("h2", event.value))

            bus.subscribe(_Foo, h1)
            bus.subscribe(_Foo, h2)
            await bus.start()
            bus.publish(_Foo(value=7))
            await bus.stop()
            return calls

        assert _run(scenario()) == [("h1", 7), ("h2", 7)]

    def test_unrelated_event_types_do_not_cross(self):
        async def scenario():
            bus = AnyIOEventBus()
            received = []

            async def foo_handler(event):
                received.append(event.value)

            bus.subscribe(_Foo, foo_handler)
            await bus.start()
            bus.publish(_Bar(value=99))
            bus.publish(_Foo(value=3))
            await bus.stop()
            return received

        assert _run(scenario()) == [3]

    def test_publish_is_non_blocking(self):
        async def scenario():
            bus = AnyIOEventBus()

            async def slow_handler(event):
                await asyncio.sleep(0.1)

            bus.subscribe(_Foo, slow_handler)
            await bus.start()
            start = time.monotonic()
            bus.publish(_Foo(value=1))
            elapsed = time.monotonic() - start
            await bus.stop()
            return elapsed

        assert _run(scenario()) < 0.05

    def test_order_is_preserved(self):
        async def scenario():
            bus = AnyIOEventBus()
            received = []

            async def handler(event):
                received.append(event.value)

            bus.subscribe(_Foo, handler)
            await bus.start()
            for value in range(1, 6):
                bus.publish(_Foo(value=value))
            await bus.stop()
            return received

        assert _run(scenario()) == [1, 2, 3, 4, 5]

    def test_handler_error_does_not_break_consumer(self):
        async def scenario():
            bus = AnyIOEventBus()
            received = []

            async def failing_handler(event):
                raise RuntimeError("boom")

            async def ok_handler(event):
                received.append(event.value)

            bus.subscribe(_Foo, failing_handler)
            bus.subscribe(_Foo, ok_handler)
            await bus.start()
            for value in range(1, 4):
                bus.publish(_Foo(value=value))
            await bus.stop()
            return received

        assert _run(scenario()) == [1, 2, 3]

    def test_stop_drains_pending_events(self):
        async def scenario():
            bus = AnyIOEventBus()
            received = []

            async def handler(event):
                await asyncio.sleep(0.01)
                received.append(event.value)

            bus.subscribe(_Foo, handler)
            await bus.start()
            bus.publish(_Foo(value=1))
            bus.publish(_Foo(value=2))
            await bus.stop()
            return received

        assert _run(scenario()) == [1, 2]

    def test_publish_after_stop_is_ignored(self):
        async def scenario():
            bus = AnyIOEventBus()
            received = []

            async def handler(event):
                received.append(event.value)

            bus.subscribe(_Foo, handler)
            await bus.start()
            bus.publish(_Foo(value=1))
            await bus.stop()
            bus.publish(_Foo(value=2))
            await asyncio.sleep(0.01)
            return received

        assert _run(scenario()) == [1]


class TestMetricsLoggingHandler:
    def test_final_only_skips_non_final_steps(self, monkeypatch):
        fake = _FakeLogger()
        monkeypatch.setattr(
            "src.simulator.infrastructure.metrics.logging.logger", fake
        )
        handler = MetricsLoggingHandler(
            policy=SamplingPolicy(mode=MetricsSamplingMode.FINAL_ONLY)
        )

        _run(handler.on_step(_step_event(step_ordinal=1)))
        assert fake.info_calls == []

        _run(handler.on_step(_step_event(step_ordinal=5)))
        assert len(fake.info_calls) == 1
        assert fake.info_calls[0][0][1] == 5

    def test_all_policy_logs_every_step(self, monkeypatch):
        fake = _FakeLogger()
        monkeypatch.setattr(
            "src.simulator.infrastructure.metrics.logging.logger", fake
        )
        handler = MetricsLoggingHandler(
            policy=SamplingPolicy(mode=MetricsSamplingMode.ALL)
        )

        for ordinal in (1, 2, 3):
            _run(handler.on_step(_step_event(step_ordinal=ordinal, total_steps=3)))

        assert [call[0][1] for call in fake.info_calls] == [1, 2, 3]

    def test_on_snapshot_without_loader_warns(self, monkeypatch):
        fake = _FakeLogger()
        monkeypatch.setattr(
            "src.simulator.infrastructure.metrics.logging.logger", fake
        )
        handler = MetricsLoggingHandler()

        _run(handler.on_snapshot(
            SimulationSnapshotPersisted(
                snapshot_id="snap",
                batch_id="batch",
                step=1,
            )
        ))

        assert len(fake.warning_calls) == 1
