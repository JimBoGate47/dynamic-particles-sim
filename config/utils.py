from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from config import DB_NAME, DOCUMENT_MODELS, DB_URI

client = AsyncIOMotorClient(DB_URI)


async def connect():
    await init_beanie(database=client[DB_NAME], document_models=DOCUMENT_MODELS)


async def disconnect():
    client.close()
