#!/bin/zsh
set -euo pipefail

cd /Users/jberman/Projects/immich-tagger

export PATH="/Users/jberman/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Load Immich credentials
source "$(dirname "$0")/.env"

echo "===== $(date) Starting daily Immich caption/tag run ====="
echo "Working directory: $(pwd)"
echo "uv path: $(command -v uv || true)"

uv run python immich_caption_and_tag.py /opt/stacks/immich-app/library/upload  \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3

echo "===== $(date) Triggering Immich sidecar sync ====="
curl -sf -X PUT "${IMMICH_URL}/api/jobs/sidecar" \
  -H "x-api-key: ${IMMICH_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"command": "start", "force": true}'
echo ""

echo "===== $(date) Finished daily Immich caption/tag run ====="
