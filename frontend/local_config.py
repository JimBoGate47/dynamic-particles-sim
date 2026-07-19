import sys
from pathlib import Path

BACKEND_PATH = str(Path(__file__).resolve().parent.parent / "backend")
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)
