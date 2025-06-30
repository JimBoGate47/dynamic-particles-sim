from abc import ABC, abstractmethod
from dataclasses import dataclass


class Interaction(ABC):
    @abstractmethod
    def compute_aceleration(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class InteractionDecorator(Interaction):
    wrapee: Interaction

    def compute_aceleration(self, *args, **kwargs):
        raise NotImplementedError
