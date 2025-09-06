import asyncio
from pprint import pprint

from backend.src.common.domain.entities import Snapshot
from backend.src.common.domain.filters.snapshot import SnapshotsFilter
from backend.src.common.infrastructure.repositories.snapshot import ORMSnapshotRepository
from backend.src.simulator.applitacion.use_cases.snapshot_lister import SnapshotsLister
from backend.src.simulator.presentation.presenters.snapshots import SnapshotsDataframePresenter
from config.database import db_connection
from plotting.plot_particles2 import plot_data


async def main():
    async with db_connection():
        snapshots: list[Snapshot] = await SnapshotsLister(
            filters=SnapshotsFilter(
                constants_name="nombre2",
            ),
            snapshot_repository=ORMSnapshotRepository(),
        ).execute()
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
            range_x=[-10, 10],
            range_y=[-10, 10],
        )


asyncio.run(main())
