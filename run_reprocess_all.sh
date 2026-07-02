#!/bin/bash
# Nightly job while the reprocess-all backfill is in progress.
# Runs 10 PM – 7 AM Pacific via cron (system clock is UTC, so cron/--stop-at
# times below are the Pacific times + 7h):
#   1. Daily incremental run first (new photos added today)
#   2. Reprocess-all backfill for the rest of the night (checkpointed via SQLite cache)
#
# Already-processed assets are skipped automatically each night.
# When the backfill completes, restore the original cron:
#   0 6 * * * /bin/bash /home/jberman/Projects/immich-tagger/run_daily_new_images.sh ...  (= 11 PM Pacific)
#
# Future: consider an Immich webhook trigger so new photos are processed within
# seconds of upload rather than waiting for the nightly cron.

set -euo pipefail

cd /home/jberman/Projects/immich-tagger

export PATH="/home/jberman/.local/bin:/usr/local/bin:/usr/bin:/bin"
source "$(dirname "$0")/.env"
export IMMICH_URL="${IMMICH_URL/http:\/\/https:\/\//https://}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/reprocess_all.log"

echo "" | tee -a "$LOG"
echo "===== $(date) Starting nightly run =====" | tee -a "$LOG"

# ── Step 1: daily incremental (new photos since last run) ──────────────────
echo "--- $(date) Daily incremental ---" | tee -a "$LOG"
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --vlm-model qwen2.5vl:7b \
  2>&1 | tee -a "$LOG"

# ── Step 2: reprocess-all backfill (runs until 7 AM Pacific, resumes tomorrow) ─
echo "--- $(date) Reprocess-all backfill (stop-at 14:00 UTC = 7 AM Pacific) ---" | tee -a "$LOG"
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --vlm-model qwen2.5vl:7b \
  --reprocess-all \
  --reprocess-captions \
  --stop-at 14:00 \
  2>&1 | tee -a "$LOG"

echo "===== $(date) Nightly run finished =====" | tee -a "$LOG"
