from dataclasses import dataclass

import torch

from backend.src.domain.interfaces import Interaction, InteractionDecorator
from backend.src.domain.entities.properties import PhysicalProperties

SECURE_DIVISION_CONSTANT = 1e-9


class PairElectrostaticInteraction(Interaction):
    def compute_aceleration(self, positions, phys_props: PhysicalProperties):
        r = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.norm(r, dim=2, keepdim=True) + SECURE_DIVISION_CONSTANT  # 1e-9 para que no haya division entre 0
        # print("DIST ", dist)
        ff = (1.0 / dist) ** 3
        aceleration = (r * ff).sum(dim=1)
        aceleration *= phys_props.q ** 2
        aceleration /= phys_props.m
        return aceleration


class BarrasInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, positions):
        pass


@dataclass
class PotencialWallInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, positions, **kwargs):
        acelerations = self.wrapee.compute_aceleration(positions, **kwargs)
        """
        acel: Tensor([ax, ay])
        pos: Tensor([x, y])
        returns: Tensor([new_ax, new_ay])
        """
        return acelerations - positions


class FrictionInteractionDecorator(InteractionDecorator):
    def compute_aceleration(self, positions, beta, velocity, **kwargs):
        aceleration = self.wrapee.compute_aceleration(positions, **kwargs)
        aceleration -= beta * velocity
        return aceleration
