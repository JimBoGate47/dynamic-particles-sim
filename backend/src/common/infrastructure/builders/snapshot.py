from src.common.domain.entities import Snapshot
from src.common.domain.models.constants import ConstantsORM
from src.common.domain.models.snapshot import SnapshotORM
from src.common.infrastructure.builders.constants import build_constants


def build_snapshot(snapshot: SnapshotORM) -> Snapshot:
    return Snapshot(
        id=str(snapshot.id),
        step=snapshot.step,
        constants=(
            build_constants(snapshot.constants)
            if isinstance(snapshot.constants, ConstantsORM) else None
        ),
        particles=snapshot.particles,
        batch_id=snapshot.batch_id,
    )
