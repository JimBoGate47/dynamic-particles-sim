from frontend.domain.types.constants import Constants
from frontend.domain.types.snapshots import Snapshot, SnapshotsCollection
from src.simulator.presentation.constants import find_constants
from src.simulator.presentation.snapshots import list_snapshots


class SimulatorService:
    async def constants_finder(self) -> list[Constants]:
        responses = await find_constants()
        return [
            Constants.model_validate(response)
            for response in responses
        ]

    async def snapshot_lister(self, constants_name: str) -> SnapshotsCollection:
        responses = await list_snapshots(constants_name)
        return SnapshotsCollection(
            snapshots=[
                Snapshot.model_validate(response)
                for response in responses
            ]
        )
