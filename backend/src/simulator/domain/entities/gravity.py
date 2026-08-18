from pydantic import BaseModel, model_validator


class GravityConfig(BaseModel):
    start: int = 1
    end: int = 10
    delta_g: float = 0.1

    @model_validator(mode="after")
    def _validate_sequence(self):
        if self.delta_g <= 0:
            raise ValueError("delta_g debe ser mayor que 0")
        if self.end < self.start:
            raise ValueError("end debe ser mayor o igual que start")
        return self
