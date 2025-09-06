from dataclasses import dataclass
from typing import Optional


@dataclass
class SnapshotsFilter:
    step: Optional[int] = None
    constants_name: Optional[str] = None
    snapshot_id: Optional[str] = None
