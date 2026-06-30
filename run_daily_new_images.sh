#!/bin/bash
set -euo pipefail

cd /home/jberman/Projects/immich-tagger

export PATH="/home/jberman/.local/bin:/usr/local/bin:/usr/bin:/bin"

# Load Immich credentials into environment for the Python script
source "$(dirname "$0")/.env"

# Fix accidental double-scheme written in .env
export IMMICH_URL="${IMMICH_URL/http:\/\/https:\/\//https://}"

echo "===== $(date) Starting daily Immich caption/tag run ====="

uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --vlm-model qwen2.5vl:7b \
  "$@"

echo "===== $(date) Finished daily Immich caption/tag run ====="
