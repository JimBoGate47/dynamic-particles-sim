from typing import Optional

import torch

from src.common.domain.entities.properties import PhysicalProps
from src.common.domain.entities.snapshot import Snapshot
from src.simulator.infrastructure.queries import GenericInteractionQuery, GenericInteractionResponse


def build_base_response(
        component: str,
        acceleration: torch.Tensor,
        mass: torch.Tensor,
        *,
        positions: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
) -> GenericInteractionResponse:
    return GenericInteractionResponse(
        positions=positions,
        velocity=velocity,
        acceleration=acceleration,
        contributions={component: (mass * acceleration).detach().clone()},
    )


def accumulate(
        response: GenericInteractionResponse,
        component: str,
        contribution: torch.Tensor,
        mass: torch.Tensor,
        *,
        positions: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
) -> GenericInteractionResponse:
    contributions = dict(response.contributions or {})
    contributions[component] = (mass * contribution).detach().clone()
    return GenericInteractionResponse(
        positions=positions if positions is not None else response.positions,
        velocity=velocity if velocity is not None else response.velocity,
        acceleration=response.acceleration + contribution,
        contributions=contributions,
    )


def apply_overrides(
        response: GenericInteractionResponse,
        *,
        positions: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
) -> GenericInteractionResponse:
    return GenericInteractionResponse(
        positions=positions if positions is not None else response.positions,
        velocity=velocity if velocity is not None else response.velocity,
        acceleration=response.acceleration,
        contributions=response.contributions,
    )


def build_query_from_snapshot(snapshot: Snapshot) -> GenericInteractionQuery:
    if snapshot.constants is None:
        raise ValueError("Snapshot has no linked constants")
    positions = torch.tensor([p.r for p in snapshot.particles], dtype=torch.float32)
    velocity = torch.tensor([p.v for p in snapshot.particles], dtype=torch.float32)
    charge = torch.tensor(
        [[p.phys_props["q"]] for p in snapshot.particles], dtype=torch.float32
    )
    mass = torch.tensor(
        [[p.phys_props["m"]] for p in snapshot.particles], dtype=torch.float32
    )
    sim_props = snapshot.constants.sim_props
    metadata_g = snapshot.metadata.get("g")
    if metadata_g is not None:
        sim_props = sim_props.model_copy(update={"g": metadata_g})
    return GenericInteractionQuery(
        positions=positions,
        velocity=velocity,
        sim_props=sim_props,
        phys_props=PhysicalProps(q=charge, m=mass),
    )
