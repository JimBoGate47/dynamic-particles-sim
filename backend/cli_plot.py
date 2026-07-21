import asyncio
from itertools import chain
from pprint import pprint

from src.common.domain.entities import Snapshot, SnapshotsCollection
from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from src.simulator.applitacion.use_cases.snapshot_lister import SnapshotsLister
from src.simulator.presentation.presenters.snapshots import SnapshotsDataframePresenter
from config.database import db_connection
from plotting.plot_particles2 import plot_data


async def main():
    async with db_connection():
        collections: list[SnapshotsCollection] = await SnapshotsLister(
            filters=SnapshotsFilter(
                constants_name="nombre3",
            ),
            snapshot_repository=ORMSnapshotRepository(),
        ).execute()
        if not collections:
            raise ValueError("No snapshots found")
        snapshots = list(chain.from_iterable(c.snapshots for c in collections))
        pprint(snapshots)

        snapshots_df = SnapshotsDataframePresenter(
            snapshots=snapshots,
        ).to_dataframe

        # TODO borrar prints
        print(snapshots_df.head())
        print(snapshots_df.tail())

        # TODO construir un plotter generico que validar columnas del dataframe
        plot_data(
            snapshots_df,
            x="rx", y="ry",
            animation_frame="step",
            hover_name="step",
            range_x=[-20, 20],
            range_y=[-20, 20],
        )


asyncio.run(main())
