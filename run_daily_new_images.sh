#!/bin/zsh
set -euo pipefail

cd /Users/jberman/Projects/immich-tagger

export PATH="/Users/jberman/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "===== $(date) Starting daily Immich caption/tag run ====="
echo "Working directory: $(pwd)"
echo "uv path: $(command -v uv || true)"

uv run python immich_caption_and_tag.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3

echo "===== $(date) Finished daily Immich caption/tag run ====="
