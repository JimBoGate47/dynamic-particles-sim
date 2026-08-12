from abc import ABC, abstractmethod
from typing import Awaitable, Callable, TypeVar

from src.common.domain.events import DomainEvent

E = TypeVar("E", bound=DomainEvent)
Handler = Callable[[DomainEvent], Awaitable[None] | None]


class UseCase(ABC):
    @abstractmethod
    async def execute(self, *args, **kwargs):
        raise NotImplementedError


class EventBus(ABC):
    """Contrato de un bus pub/sub in-process de eventos de dominio."""

    @abstractmethod
    def subscribe(self, event_type: type[E], handler: Handler) -> None:
        raise NotImplementedError

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError
