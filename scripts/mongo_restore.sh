#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${MONGO_CONTAINER:-sim-anillos-mongo}"

if [ "$#" -lt 1 ]; then
  echo "uso: $0 <backup.archive.gz> [--drop]" >&2
  echo "  --drop  elimina las colecciones existentes antes de restaurar" >&2
  exit 1
fi

backup_file="$1"
shift

if [ ! -f "$backup_file" ]; then
  echo "error: no se encuentra el archivo '$backup_file'" >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "error: el contenedor '$CONTAINER' no está corriendo" >&2
  exit 1
fi

args=(--archive --gzip)
for arg in "$@"; do
  if [ "$arg" = "--drop" ]; then
    args+=(--drop)
  fi
done

docker exec -i "$CONTAINER" mongorestore "${args[@]}" < "$backup_file"

echo "Restore completado desde $backup_file"