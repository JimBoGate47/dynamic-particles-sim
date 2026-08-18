from loguru import logger

from frontend.domain.enums import ConfinementType
from frontend.domain.types.constants import Constants
from frontend.domain.types.gravity import GravityConfig
from frontend.domain.types.snapshots import Snapshot, SnapshotsCollection
from src.simulator.presentation.constants import (
    CreateConstantsRequest,
    create_constants,
    find_constants,
)
from src.simulator.presentation.snapshots import (
    CreateSnapshotRequest,
    GetSnapshotRequest,
    ListSnapshotsRequest,
    RunSimulationRequest,
    RunSimulationWithGravityRequest,
    create_snapshot,
    get_snapshot,
    list_snapshots,
    run_simulation,
    run_simulation_with_gravity,
)


class SimulatorService:
    async def constants_finder(self) -> list[Constants]:
        responses = await find_constants()
        return [
            Constants.model_validate(response)
            for response in responses
        ]

    async def snapshot_lister(self, constants_name: str) -> list[SnapshotsCollection]:
        logger.info("Fetching snapshots for {}", constants_name)
        responses = await list_snapshots(ListSnapshotsRequest(constants_name=constants_name))
        logger.info("Received {} collections", len(responses))

        return [
            SnapshotsCollection.model_validate(response)
            for response in responses
        ]

    async def snapshot_finder(self, snapshot_id: str) -> Snapshot:
        logger.info("Fetching snapshot {}", snapshot_id)
        response = await get_snapshot(GetSnapshotRequest(snapshot_id=snapshot_id))
        return Snapshot.model_validate(response)

    async def snapshot_creator(self, data: dict) -> Snapshot:
        logger.info("Creating snapshot with step={}", data.get("step"))
        response = await create_snapshot(CreateSnapshotRequest(**data))
        return Snapshot.model_validate(response)

    async def simulation_runner(
        self,
        snapshot_id: str,
        n_steps: int = 506,
        save_at_mod: int = 100,
        wall: ConfinementType = ConfinementType.HARMONIC,
    ) -> list[Snapshot]:
        logger.info("Running simulation from snapshot {}", snapshot_id)
        response = await run_simulation(RunSimulationRequest(
            snapshot_id=snapshot_id,
            n_steps=n_steps,
            save_at_mod=save_at_mod,
            wall=wall,
        ))
        return [Snapshot.model_validate(s) for s in response]

    async def simulation_plus_gravity_runner(
        self,
        snapshot_id: str,
        stabilization_steps: int = 506,
        gravity_config: GravityConfig | None = None,
        wall: ConfinementType = ConfinementType.HARMONIC,
    ) -> list[Snapshot]:
        request = RunSimulationWithGravityRequest(
            snapshot_id=snapshot_id,
            stabilization_steps=stabilization_steps,
            wall=wall,
        )

        if gravity_config is not None:
            request.gravity_config = gravity_config

        logger.info("Running gravity simulation from snapshot {} request: {}", snapshot_id, request)
        response = await run_simulation_with_gravity(request)
        return [Snapshot.model_validate(s) for s in response]

    async def constants_creator(self, data: dict) -> Constants:
        logger.info("Creating constants with name={}", data.get("name"))
        response = await create_constants(CreateConstantsRequest(**data))
        return Constants.model_validate(response)
