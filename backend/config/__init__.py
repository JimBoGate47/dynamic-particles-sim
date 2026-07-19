from src.common.domain.models.constants import ConstantsORM
from src.common.domain.models.snapshot import SnapshotORM

DB_URI = "mongodb://localhost:27017"
DB_NAME = "anillos"
DOCUMENT_MODELS = [
    ConstantsORM,
    SnapshotORM,
]
