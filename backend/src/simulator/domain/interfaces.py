from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

Q = TypeVar("Q", bound="InteractionQuery")
R = TypeVar("R", bound="InteractionResponse")


class Interaction(ABC, Generic[Q, R]):
    @abstractmethod
    def compute_aceleration(self, query: Q) -> R:
        raise NotImplementedError


@dataclass
class InteractionDecorator(Interaction[Q, R], Generic[Q, R]):
    _wrapee: Interaction[Q, R]

    @abstractmethod
    def compute_aceleration(self, query: Q) -> R:
        return self._wrapee.compute_aceleration(query)
