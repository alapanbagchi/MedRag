#!/usr/bin/env bash
# ============================================================
# MedRAG pgvector setup — run this from your real terminal
# ============================================================
set -euo pipefail

CONTAINER="medrag-pgvector"
PORT="${PGPORT:-5432}"
USER="${PGUSER:-postgres}"
PASS="${PGPASSWORD:-medrag}"
DB="${PGDATABASE:-medrag}"
IMG="pgvector/pgvector:pg16"

echo "============================================"
echo "  MedRAG pgvector setup"
echo "============================================"

# 1. Stop any existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Stopping existing container..."
    docker rm -f "$CONTAINER" >/dev/null 2>&1
fi

# 2. Start the container
echo "Starting $IMG on port $PORT..."
docker run -d \
    --name "$CONTAINER" \
    -p "${PORT}:5432" \
    -e POSTGRES_USER="$USER" \
    -e POSTGRES_PASSWORD="$PASS" \
    -e POSTGRES_DB="$DB" \
    "$IMG"

# 3. Wait for healthy
echo "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker exec "$CONTAINER" pg_isready -U "$USER" -q 2>/dev/null; then
        echo "  Ready!"
        break
    fi
    sleep 1
done

# 4. Verify pgvector
echo "Verifying pgvector extension..."
docker exec "$CONTAINER" psql -U "$USER" -d "$DB" -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

echo ""
echo "============================================"
echo "  Done! PostgreSQL + pgvector is running."
echo "============================================"
echo ""
echo "Connection details:"
echo "  Host:     localhost"
echo "  Port:     $PORT"
echo "  User:     $USER"
echo "  Password: $PASS"
echo "  Database: $DB"
echo ""
echo "Env vars for MedRAG:"
echo "  export PGHOST=localhost"
echo "  export PGPORT=$PORT"
echo "  export PGUSER=$USER"
echo "  export PGPASSWORD=$PASS"
echo "  export PGDATABASE=$DB"
echo ""
echo "Next step:"
echo "  python scripts/init_pgvector.py --load-from embeddings --corpus index/corpus.parquet"
