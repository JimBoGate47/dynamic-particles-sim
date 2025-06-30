from abc import ABC, abstractmethod


class Particles2DRepository(ABC):
    @abstractmethod
    def filter(self):
        raise NotImplementedError

    @abstractmethod
    def persist(self):
        raise NotImplementedError
