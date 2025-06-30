from typing import List, Optional

from pydantic import BaseModel


class ParticleORM(BaseModel):
    r: List[float]
    v: List[float]
    a: List[float]
    phys_props: Optional[dict] = {}

    def to_json(self) -> dict:
        return {
            "r": self.r,
            "v": self.v,
            "a": self.a,
            "phys_props": self.props,
        }
