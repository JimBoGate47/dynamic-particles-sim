from dataclasses import dataclass

import torch

from backend.src.simulator.domain.interfaces import Interaction, InteractionDecorator
from backend.src.simulator.infrastructure.queries import GenericInteractionQuery

SECURE_DIVISION_CONSTANT = 1e-9


class PairElectrostaticInteraction(Interaction):
    def compute_aceleration(self, query: GenericInteractionQuery, **kwargs):
        r = query.positions.unsqueeze(1) - query.positions.unsqueeze(0)
        dist = torch.norm(r, dim=2, keepdim=True) + SECURE_DIVISION_CONSTANT  # 1e-9 para que no haya division entre 0
        # print("DIST ", dist)
        ff = (1.0 / dist) ** 3
        aceleration = (r * ff).sum(dim=1)
        aceleration *= query.phys_props.q ** 2
        aceleration /= query.phys_props.m
        return aceleration


class BarrasInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, positions):
        pass


@dataclass
class PotencialWallInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, query: GenericInteractionQuery, **kwargs):
        acelerations = self.wrapee.compute_aceleration(query, **kwargs)
        """
        acel: Tensor([ax, ay])
        pos: Tensor([x, y])
        returns: Tensor([new_ax, new_ay])
        """
        return acelerations - query.positions


class FrictionInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, query: GenericInteractionQuery, **kwargs):
        aceleration = self.wrapee.compute_aceleration(query)
        aceleration -= query.beta * query.velocity
        return aceleration
