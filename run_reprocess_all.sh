#!/bin/bash
# Backfill job — runs continuously (no time-of-day cutoff) until the
# reprocess-all backfill has covered every asset in the library.
#   1. Daily incremental run first (new photos added today)
#   2. Reprocess-all backfill, uninterrupted, checkpointed via SQLite cache
#
# Still fires nightly via cron (0 5 * * *), but a flock guard makes that safe:
# if a previous run is still going (which is expected while the no-cutoff
# backfill is in progress), the new cron invocation just exits instead of
# starting a second overlapping process.
#
# When the backfill completes, restore the original cron:
#   0 6 * * * /bin/bash /home/jberman/Projects/immich-tagger/run_daily_new_images.sh ...  (= 11 PM Pacific)
#
# Future: consider an Immich webhook trigger so new photos are processed within
# seconds of upload rather than waiting for the nightly cron.

set -euo pipefail

cd /home/jberman/Projects/immich-tagger

exec 200>/tmp/immich_reprocess_all.lock
if ! flock -n 200; then
  echo "$(date) Previous run still in progress — skipping this invocation." >> logs/reprocess_all.log
  exit 0
fi

export PATH="/home/jberman/.local/bin:/usr/local/bin:/usr/bin:/bin"
source "$(dirname "$0")/.env"
export IMMICH_URL IMMICH_API_KEY

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/reprocess_all.log"

echo "" | tee -a "$LOG"
echo "===== $(date) Starting run =====" | tee -a "$LOG"

# ── Step 1: daily incremental (new photos since last run) ──────────────────
echo "--- $(date) Daily incremental ---" | tee -a "$LOG"
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --vlm-model qwen2.5vl:7b \
  2>&1 | tee -a "$LOG"

# ── Step 2: reprocess-all backfill (runs uninterrupted until fully caught up) ─
echo "--- $(date) Reprocess-all backfill (no time cutoff) ---" | tee -a "$LOG"
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --vlm-model qwen2.5vl:7b \
  --reprocess-all \
  --reprocess-captions \
  2>&1 | tee -a "$LOG"

echo "===== $(date) Run finished =====" | tee -a "$LOG"
