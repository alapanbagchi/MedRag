#!/usr/bin/env bash
# ============================================================
# Start the MedRAG pgvector PostgreSQL container.
#
# Idempotent: if the container already exists (running or
# stopped) it is started; otherwise it is created. Data is
# never destroyed, unlike setup_pgvector.sh which re-creates
# the container from scratch.
#
# Override with env: CONTAINER, PGPORT, PGUSER, PGPASSWORD,
# PGDATABASE, PG_IMAGE.
# ============================================================
set -euo pipefail

CONTAINER="${CONTAINER:-medrag-pgvector}"
PORT="${PGPORT:-5432}"
USER="${PGUSER:-postgres}"
PASS="${PGPASSWORD:-medrag}"
DB="${PGDATABASE:-medrag}"
IMG="${PG_IMAGE:-pgvector/pgvector:pg16}"

echo "============================================"
echo "  MedRAG pgvector container"
echo "============================================"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "  Container '$CONTAINER' is already running."
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "  Starting existing container '$CONTAINER'..."
    docker start "$CONTAINER" >/dev/null
else
    echo "  Creating container '$CONTAINER' from $IMG..."
    docker run -d \
        --name "$CONTAINER" \
        -p "${PORT}:5432" \
        -e POSTGRES_USER="$USER" \
        -e POSTGRES_PASSWORD="$PASS" \
        -e POSTGRES_DB="$DB" \
        "$IMG" >/dev/null
fi

echo "  Waiting for PostgreSQL to be ready..."
for i in $(seq 1 60); do
    if docker exec "$CONTAINER" pg_isready -U "$USER" -q 2>/dev/null; then
        echo "  Ready!"
        break
    fi
    sleep 1
    if [ "$i" = 60 ]; then
        echo "  ERROR: PostgreSQL did not become ready." >&2
        docker logs --tail 20 "$CONTAINER" >&2 || true
        exit 1
    fi
done

echo "  Ensuring pgvector extension..."
docker exec "$CONTAINER" psql -U "$USER" -d "$DB" -v ON_ERROR_STOP=1 \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo ""
echo "  Done. Connection: ${PGHOST:-localhost}:${PORT}/${DB} (user ${USER})"
echo "  Data dir inside container persists until 'docker rm $CONTAINER'."
echo ""
echo "  Next: make pg-load   (or: make pg-load RESET=1 to wipe + reload)"