from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from src.common.domain.entities.properties import PhysicalProps, SimulationProps
from src.common.domain.entities.snapshot import Snapshot
from src.simulator.domain.interfaces import Interaction
from src.simulator.infrastructure.helpers.interaction import build_query_from_snapshot
from src.simulator.infrastructure.queries import (
    GenericInteractionResponse,
)

SECURE_DIVISION_CONSTANT = 1e-9


# def _clone(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#     if tensor is None:
#         return None
#     return tensor.detach().clone()


def mean_speed(velocity: torch.Tensor) -> float:
    return float(torch.linalg.norm(velocity, dim=1).mean())


def rms_speed(velocity: torch.Tensor) -> float:
    return float(torch.linalg.norm(velocity, dim=1).pow(2).mean().sqrt())


def kinetic_energy(mass: torch.Tensor, velocity: torch.Tensor) -> float:
    return float((0.5 * mass * (velocity ** 2).sum(dim=1, keepdim=True)).sum())


def temperature(mass: torch.Tensor, velocity: torch.Tensor) -> float:
    degrees_of_freedom = 2 * mass.shape[0]
    if degrees_of_freedom == 0:
        return 0.0
    return kinetic_energy(mass, velocity) / degrees_of_freedom


def coulomb_energy(charge: torch.Tensor, k: float, positions: torch.Tensor) -> float:
    r = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = torch.norm(r, dim=2) + SECURE_DIVISION_CONSTANT
    charge_pair = (charge @ charge.transpose(0, 1)).squeeze(-1)
    energy_matrix = k * charge_pair / dist
    mask = ~torch.eye(charge.shape[0], dtype=torch.bool, device=charge.device)
    return float(energy_matrix[mask].sum() / 2.0)


@dataclass
class SimulationMetrics:
    step: int
    forces: dict[str, torch.Tensor]
    aggregates: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            # "forces": {
            #     name: force.tolist() for name, force in self.forces.items()
            # },
            "aggregates": self.aggregates,
        }


def build_metrics(
        *,
        step: int,
        positions: torch.Tensor,
        velocity: torch.Tensor,
        phys_props: PhysicalProps,
        sim_props: SimulationProps,
        response: GenericInteractionResponse,
) -> SimulationMetrics:
    forces = {
        name: torch.linalg.norm(force, dim=1)
        for name, force in (response.contributions or {}).items()
    }
    # TODO revisar que las metricas son correctas
    aggregates = {
        "mean_speed": mean_speed(velocity),
        "rms_speed": rms_speed(velocity),
        "kinetic_energy": kinetic_energy(phys_props.m, velocity),
        "temperature": temperature(phys_props.m, velocity),
        "coulomb_energy": coulomb_energy(phys_props.q, sim_props.k, positions),
    }
    for name, force in forces.items():
        aggregates[f"{name}.min_force"] = float(force.min())
        aggregates[f"{name}.mean_force"] = float(force.mean())
        aggregates[f"{name}.max_force"] = float(force.max())
    return SimulationMetrics(step=step, forces=forces, aggregates=aggregates)


class MetricsEngine:
    @staticmethod
    def compute(snapshot: Snapshot, interactions: Interaction) -> SimulationMetrics:
        query = build_query_from_snapshot(snapshot)
        response = interactions.compute_aceleration(query)
        return build_metrics(
            step=snapshot.step,
            positions=query.positions,
            velocity=query.velocity,
            phys_props=query.phys_props,
            sim_props=query.sim_props,
            response=response,
        )

    @staticmethod
    def compute_from_response(
            *,
            step: int,
            positions: torch.Tensor,
            velocity: torch.Tensor,
            phys_props: PhysicalProps,
            sim_props: SimulationProps,
            response: GenericInteractionResponse,
    ) -> SimulationMetrics:
        return build_metrics(
            step=step,
            positions=positions,
            velocity=velocity,
            phys_props=phys_props,
            sim_props=sim_props,
            response=response,
        )


class MetricsSamplingMode(str, Enum):
    FINAL_ONLY = "final_only"
    ALL = "all"
    EVERY_N = "every_n"

    @property
    def is_final_only(self) -> bool:
        return self == self.FINAL_ONLY

    @property
    def is_all(self) -> bool:
        return self == self.ALL

    @property
    def is_every_n(self) -> bool:
        return self == self.EVERY_N


@dataclass
class SamplingPolicy:
    mode: MetricsSamplingMode = MetricsSamplingMode.FINAL_ONLY
    every_n: int = 1

    def should_capture(self, step: int, total_steps: int) -> bool:
        if self.mode == MetricsSamplingMode.FINAL_ONLY:
            return step == total_steps
        if self.mode == MetricsSamplingMode.EVERY_N:
            return step % max(self.every_n, 1) == 0 or step == total_steps
        return True


def compute_metrics_for_snapshots(
        snapshots: list[Snapshot],
        interactions: Interaction,
        policy: SamplingPolicy,
) -> list[SimulationMetrics]:
    snapshots = sorted(snapshots, key=lambda s: s.step)
    total = len(snapshots)
    metrics = []
    for index, snapshot in enumerate(snapshots, start=1):
        if policy.should_capture(index, total):
            metrics.append(MetricsEngine.compute(snapshot, interactions))
    return metrics
