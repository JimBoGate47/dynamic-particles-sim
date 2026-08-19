import io
import json
import zipfile
from dataclasses import dataclass

from src.common.domain.filters.snapshot import SnapshotsFilter
from src.common.domain.interfaces import UseCase
from src.common.domain.repositories.snapshot import SnapshotRepository


@dataclass
class SnapshotBatchZip:
    filename: str
    content: bytes


@dataclass
class SnapshotsBatchZipper(UseCase):
    batch_id: str
    snapshot_repository: SnapshotRepository

    async def execute(self, *args, **kwargs) -> SnapshotBatchZip | None:
        snapshots = await self.snapshot_repository.filter(
            params=SnapshotsFilter(batch_id=self.batch_id),
        )
        if not snapshots:
            return None

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for snapshot in sorted(snapshots, key=lambda s: s.step):
                zf.writestr(
                    f"snapshot_{snapshot.step}.json",
                    json.dumps(snapshot.model_dump(mode="json"), indent=2),
                )

        return SnapshotBatchZip(
            filename=f"snapshot_{self.batch_id}.zip",
            content=buffer.getvalue(),
        )