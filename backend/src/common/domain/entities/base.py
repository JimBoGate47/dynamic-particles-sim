from bson import ObjectId
from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    id: str | None = None

    @property
    def id_object(self) -> ObjectId:
        return ObjectId(self.id)
