from collections import defaultdict
from inspect import isawaitable
from typing import TypeVar

from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectReceiveStream
from loguru import logger

from src.common.domain.events import DomainEvent
from src.common.domain.interfaces import EventBus, Handler

E = TypeVar("E", bound=DomainEvent)


class AnyIOEventBus(EventBus):
    """Implementación de :class:`EventBus` con memory object stream (AnyIO) + task group.

    ``publish`` es no bloqueante (``send_nowait`` sobre buffer no acotado): la
    simulación solo encola y nunca espera a los handlers. Un único consumidor
    (``_consume``) procesa los eventos en orden FIFO. ``stop`` cierra el send
    stream, lo que hace que el consumidor drene lo pendiente y termine con
    ``EndOfStream`` (shutdown ordenado garantizado por el task group).
    """

    def __init__(self) -> None:
        self._send, self._recv = create_memory_object_stream[DomainEvent](float("inf"))
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._task_group = None
        self._closed = False

    def subscribe(self, event_type: type[E], handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: DomainEvent) -> None:
        if self._closed:
            return
        self._send.send_nowait(event)

    async def start(self) -> None:
        if self._task_group is not None:
            return
        self._task_group = create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._consume, self._recv)

    async def stop(self) -> None:
        self._closed = True
        if self._task_group is None:
            return
        await self._send.aclose()
        await self._task_group.__aexit__(None, None, None)
        self._task_group = None

    async def _consume(self, recv: MemoryObjectReceiveStream[DomainEvent]) -> None:
        async with recv:
            async for event in recv:
                for handler in list(self._handlers[type(event)]):
                    try:
                        result = handler(event)
                        if isawaitable(result):
                            await result
                    except Exception:
                        logger.exception(
                            "Event handler failed for {}", type(event).__name__
                        )
