from loguru import logger

from frontend.domain.types.constants import Constants
from frontend.domain.types.snapshots import Snapshot, SnapshotsCollection
from src.simulator.presentation.constants import create_constants, find_constants
from src.simulator.presentation.snapshots import create_snapshot, get_snapshot, list_snapshots


class SimulatorService:
    async def constants_finder(self) -> list[Constants]:
        responses = await find_constants()
        return [
            Constants.model_validate(response)
            for response in responses
        ]

    async def snapshot_lister(self, constants_name: str) -> list[SnapshotsCollection]:
        logger.info("Fetching snapshots for {}", constants_name)
        responses = await list_snapshots(constants_name)
        logger.info("Received {} collections", len(responses))

        return [
            SnapshotsCollection.model_validate(response)
            for response in responses
        ]

    async def snapshot_finder(self, snapshot_id: str) -> Snapshot:
        logger.info("Fetching snapshot {}", snapshot_id)
        response = await get_snapshot(snapshot_id)
        return Snapshot.model_validate(response)

    async def snapshot_creator(self, data: dict) -> Snapshot:
        logger.info("Creating snapshot with step={}", data.get("step"))
        response = await create_snapshot(data)
        return Snapshot.model_validate(response)

    async def constants_creator(self, data: dict) -> Constants:
        logger.info("Creating constants with name={}", data.get("name"))
        response = await create_constants(data)
        return Constants.model_validate(response)
