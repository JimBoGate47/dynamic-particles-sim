from dataclasses import dataclass

import torch

from backend.src.simulator.domain.interfaces import SystemRestriction
from backend.src.simulator.infrastructure.queries import PositionRestrictionResponse, PositionRestrictionQuery


@dataclass
class IndexRestrictionParams:
    idx_i: torch.Tensor
    idx_j: torch.Tensor

    @classmethod
    def from_list(cls, indices: list[tuple[int, int]], device) -> "IndexRestrictionParams":
        return cls(
            idx_i=torch.tensor([i for i, j in indices], device=device),
            idx_j=torch.tensor([i for i, j in indices], device=device),
        )


@dataclass
class InPlacePositionSystemRestriction(SystemRestriction[PositionRestrictionQuery, PositionRestrictionResponse]):
    index_restriction: IndexRestrictionParams

    def apply(
            self,
            query: PositionRestrictionQuery,
    ) -> PositionRestrictionResponse:
        """
            Returns new_positions where, for each index (i, j),
            the value from old_positions[i, j] is restored.
        """
        query.new_positions[
            self.index_restriction.idx_i,
            self.index_restriction.idx_j
        ] = query.old_positions[
            self.index_restriction.idx_i,
            self.index_restriction.idx_j
        ]
        return PositionRestrictionResponse(
            new_positions=query.new_positions,
        )
