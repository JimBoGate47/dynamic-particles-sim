from dataclasses import dataclass

import pandas as pd

from src.common.domain.entities import Snapshot


@dataclass
class SnapshotsDataframePresenter:
    snapshots: list[Snapshot]

    @property
    def to_dataframe(self) -> pd.DataFrame:
        evolucion = []

        for snapshot in self.snapshots:
            evolucion += snapshot.export_particles()

        return pd.DataFrame(evolucion)
