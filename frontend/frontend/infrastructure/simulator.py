import asyncio

from frontend.domain.types.constants import Constants
from src.simulator.presentation.constants import find_constants


class SimulatorService:
    async def constants_finder(self) -> list[Constants]:
        responses = await find_constants()
        return [
            Constants.model_validate(response)
            for response in responses
        ]


if __name__ == '__main__':
    service = SimulatorService()
    res = asyncio.run(service.constants_finder())
    print(res)
