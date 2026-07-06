# CLAUDE.md

**`README.md` is the authoritative doc** — master file directory, installation, full CLI
parameter reference, models/prompts, tag-hierarchy design, and the update/re-tag workflow.
Read it before making changes; keep it updated when behavior changes.

## Quick orientation

- v2, API-based: everything goes through the Immich HTTP API. No sidecar files. The v1
  sidecar toolkit was fully removed 2026-07-06 — no v1 code exists anywhere.
- Entry point: `immich_caption_and_tag_v2.py`, normally invoked via `run_daily_new_images.sh`
  (incremental) or `run_reprocess_all.sh` (full backfill). Both pass
  `--vlm-url http://localhost:8082 --vlm-api openai`.
- Captions come from the `llama-vlm` systemd service (llama.cpp + Vulkan on the Radeon 890M
  iGPU, Qwen2.5-VL 7B). Check with `curl localhost:8082/health` / `journalctl -u llama-vlm`.
  Plain Ollama on :11434 also exists but is not used by the run scripts (CPU vision encoder
  is ~2× slower).
- Python via `uv` (`uv run python ...`), Python ≥ 3.14. SigLIP runs on CPU — torch has no
  ROCm build here, so `--device auto` → cpu is expected, not a bug.

## Live state — handle with care

- `.immich_tagger_v2_cache.sqlite3` is live pipeline state: per-asset model-sig cache
  (resume/skip mechanism) plus the incremental last-run bookmark. Deleting it forces a
  full reprocess of ~19k assets (days of runtime).
- `.env` holds `IMMICH_URL` / `IMMICH_API_KEY` — never commit.
- A long-running tagger process may be active at any time. Before starting a run that
  writes, check `flock -n /tmp/immich_reprocess_all.lock true` or look for a running
  `immich_caption_and_tag_v2.py` process.
- Scheduling lives in the **host crontab only** (`crontab -l`), not in this repo: daily
  incremental at 06:00 UTC (flock-guarded), `@reboot` backfill safety net, hourly
  `health_snapshot.py`. Check `logs/health.log` for pipeline liveness
  (`imgs_last_hour=0` during an active run means something is stuck).
