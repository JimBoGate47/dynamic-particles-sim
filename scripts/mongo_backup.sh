#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${MONGO_CONTAINER:-sim-anillos-mongo}"
DB_NAME="${MONGO_DB:-anillos}"
OUT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)/backups"

timestamp="$(date +%Y%m%d_%H%M%S)"
out_file="$OUT_DIR/${DB_NAME}_${timestamp}.archive.gz"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "error: el contenedor '$CONTAINER' no está corriendo" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

docker exec "$CONTAINER" mongodump --db "$DB_NAME" --archive --gzip > "$out_file"

echo "Backup creado: $out_file ($(du -h "$out_file" | cut -f1))"