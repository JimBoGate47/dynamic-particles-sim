from src.common.domain.entities import Snapshot, SnapshotsCollection


def group_by_batch_id(snapshots: list[Snapshot]) -> list[SnapshotsCollection]:
    groups: dict[str, list[Snapshot]] = {}
    for snap in snapshots:
        groups.setdefault(snap.batch_id, []).append(snap)
    return [
        SnapshotsCollection(batch_id=bid, snapshots=group)
        for bid, group in groups.items()
    ]